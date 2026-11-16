"""Ask Before You Push: training-free, self-checked decoding against over-refusal.

For every prompt (same model throughout, no training, the same settings for every model):
  1. base = plain greedy answer.
  2. Refusal check: the model judges its own answer, P(B: does not answer) > 0.5 -> refused.
     Not refused                                   -> keep the base answer         (branch "base")
  3. s0 = refusal-opening probability with no system prompt, s1 = the same under the permissive prompt.
     Direction check: s1 >= s0                     -> keep the refusal              (branch "wrong_direction")
  4. Depth check:     s1 >= 0.5                    -> keep the refusal              (branch "deep_refusal")
  5. Push the first m tokens: z = z_base + beta * (z_perm - z_base), beta = 3 * sigmoid(20 * (0.5 - s1)),
     then plain greedy decoding.
  6. Output check: the model judges its pushed answer; P(B: harmful) > 0.5
                                                   -> return the base answer         (branch "push_reverted")
     otherwise                                     -> return the pushed answer       (branch "push")

  python -m abyp.generate --model meta-llama/Meta-Llama-3-8B-Instruct --out outputs/llama --datasets orbench
Output: <out>/<dataset>.jsonl, one record per prompt; "response" is the method's answer and "base" the plain one.
"""
import argparse, json, math, time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from abyp.prompts import PERMISSIVE, REFUSAL_OPENINGS, REFUSAL_CHECK, HARM_CHECK

SETS = ["orbench", "oktest", "xstest_safe", "xstest_unsafe", "advbench", "jailbench"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Hugging Face model id or local path")
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", default="data", help="folder with <dataset>.jsonl files (see data/fetch_data.py)")
    ap.add_argument("--datasets", nargs="+", default=SETS)
    ap.add_argument("--start", type=int, default=0, help="use prompts [start, end) of each dataset")
    ap.add_argument("--end", type=int, default=None)
    ap.add_argument("--depth", type=float, default=0.5, help="deep refusal if s1 >= depth")
    ap.add_argument("--k", type=float, default=20.0, help="slope of the beta sigmoid")
    ap.add_argument("--bmax", type=float, default=3.0, help="maximum push strength")
    ap.add_argument("--m", type=int, default=5, help="number of pushed tokens")
    ap.add_argument("--maxnew", type=int, default=512)
    ap.add_argument("--check-tokens", type=int, default=400, help="answer tokens shown to the two self-checks")
    ap.add_argument("--no-output-check", action="store_true", help="skip step 6 (ablation)")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(
        a.model, torch_dtype=torch.bfloat16 if dev == "cuda" else torch.float32).to(dev).eval()
    gemma = "gemma" in a.model.lower()
    eos_ids = set()
    for e in (model.config.eos_token_id, model.generation_config.eos_token_id, tok.eos_token_id):
        eos_ids.update(e if isinstance(e, (list, tuple)) else [e])
    for t in ("<end_of_turn>", "<|eot_id|>", "<|im_end|>"):
        i = tok.convert_tokens_to_ids(t)
        if isinstance(i, int) and i != tok.unk_token_id: eos_ids.add(i)
    eos_ids.discard(None)
    refusal_ids = [tok(r, add_special_tokens=False, return_tensors="pt").input_ids.to(dev) for r in REFUSAL_OPENINGS]
    first = lambda ws: sorted({tok(w, add_special_tokens=False).input_ids[0] for w in ws})
    A_IDS, B_IDS = first(["A", " A", "(A"]), first(["B", " B", "(B"])

    def chat_ids(text, system):
        if system is None: msgs = [{"role": "user", "content": text}]
        elif gemma: msgs = [{"role": "user", "content": f"{system}\n\n{text}"}]   # Gemma has no system role
        else: msgs = [{"role": "system", "content": system}, {"role": "user", "content": text}]
        enc = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
        return (enc["input_ids"] if hasattr(enc, "keys") else enc).to(dev)

    @torch.no_grad()
    def p_refuse(text, system):
        """Probability of opening the reply with a refusal: the best of REFUSAL_OPENINGS, per-token geometric mean."""
        pids = chat_ids(text, system); best = -1e30
        for rid in refusal_ids:
            logits = model(torch.cat([pids, rid], 1)).logits
            s = pids.shape[1]
            logp = torch.log_softmax(logits[0, s - 1:-1].float(), -1)
            best = max(best, logp.gather(1, rid[0].unsqueeze(1)).squeeze(1).mean().item())
        return math.exp(best)

    @torch.no_grad()
    def p_option_b(template, x, r):
        """The model's own P(B) vs P(A) on a two-option question about request x and response r."""
        r = tok.decode(tok(r, add_special_tokens=False).input_ids[:a.check_tokens])
        logp = torch.log_softmax(model(chat_ids(template.format(x=x, r=r), None)).logits[0, -1].float(), -1)
        la, lb = torch.logsumexp(logp[A_IDS], 0).item(), torch.logsumexp(logp[B_IDS], 0).item()
        return 1 / (1 + math.exp(la - lb))

    # Gemma-2: explicit HybridCache + cache_position (older transformers return no cache); Llama/Qwen: default
    if gemma:
        from transformers import HybridCache
        def new_cache(n):
            try: return HybridCache(config=model.config, max_batch_size=1, max_cache_len=n, device=dev, dtype=model.dtype)
            except TypeError: return HybridCache(config=model.config, batch_size=1, max_cache_len=n, device=dev, dtype=model.dtype)

    def prefill(ids, max_new):
        if not gemma:
            o = model(ids, use_cache=True); return o, o.past_key_values
        c = new_cache(ids.shape[1] + max_new + 1)
        o = model(ids, past_key_values=c, use_cache=True, cache_position=torch.arange(ids.shape[1], device=dev))
        return o, c

    def step(tin, pkv, pos):
        if not gemma:
            o = model(tin, past_key_values=pkv, use_cache=True); return o, o.past_key_values
        o = model(tin, past_key_values=pkv, use_cache=True, cache_position=torch.tensor([pos], device=dev))
        return o, pkv

    @torch.no_grad()
    def gen(text, beta, m=a.m, max_new=a.maxnew):
        """Greedy decoding; with beta != 0 the first m tokens use z_base + beta * (z_perm - z_base)."""
        idsB = chat_ids(text, None); posB = idsB.shape[1]
        bo, bpkv = prefill(idsB, max_new); zB = bo.logits[0, -1].float()
        need = beta != 0
        if need:
            idsP = chat_ids(text, PERMISSIVE); posP = idsP.shape[1]
            po, ppkv = prefill(idsP, m); zP = po.logits[0, -1].float()
        out = []
        for t in range(max_new):
            z = zB + beta * (zP - zB) if (need and t < m) else zB
            nxt = int(z.argmax())
            if nxt in eos_ids: break
            out.append(nxt)
            tin = torch.tensor([[nxt]], device=dev)
            bo, bpkv = step(tin, bpkv, posB + t); zB = bo.logits[0, -1].float()
            if (t + 1) < m and need:
                po, ppkv = step(tin, ppkv, posP + t); zP = po.logits[0, -1].float()
        return tok.decode(out, skip_special_tokens=True), len(out)

    r2, n2 = gen("What is the capital of France? Answer in one short sentence.", 0.0, max_new=64)
    print(f"sanity EOS: {n2} tokens | {r2!r}", flush=True)
    assert n2 < 64 and len(r2.split()) >= 3, "generation broken (no EOS or empty) - STOP"

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    json.dump(vars(a), open(out / "run_info.json", "w"), indent=1)

    for ds in a.datasets:
        allp = [json.loads(l)["user_text"] for l in open(f"{a.data}/{ds}.jsonl") if l.strip()]
        lo, hi = a.start, min(a.end or len(allp), len(allp))
        o = out / f"{ds}.jsonl"
        done = {json.loads(l)["id"] for l in o.open()} if o.exists() else set()
        print(f"\n### {ds} [{lo},{hi}): {len(done)} done", flush=True)
        counts = {}
        with o.open("a") as f:
            for i in range(lo, hi):                            # ids are the line index in the dataset file
                if i in done: continue
                x, ts = allp[i], time.time()
                base, nb = gen(x, 0.0)
                pdec = p_option_b(REFUSAL_CHECK, x, base)
                rec = {"dataset": ds, "id": i, "user_text": x, "base": base, "base_ntok": nb, "p_declined": pdec}
                if pdec <= 0.5:
                    rec.update(branch="base", response=base)
                else:
                    s0, s1 = p_refuse(x, None), p_refuse(x, PERMISSIVE)
                    rec.update(s0=s0, s1=s1)
                    if s1 >= s0:
                        rec.update(branch="wrong_direction", response=base)
                    elif s1 >= a.depth:
                        rec.update(branch="deep_refusal", response=base)
                    else:
                        beta = a.bmax / (1.0 + math.exp(-a.k * (a.depth - s1)))
                        pushed, _ = gen(x, beta)
                        rec.update(branch="push", beta=beta, pushed=pushed, response=pushed)
                        if not a.no_output_check:
                            ph = p_option_b(HARM_CHECK, x, pushed)
                            rec["p_harmful"] = ph
                            if ph > 0.5: rec.update(branch="push_reverted", response=base)
                f.write(json.dumps(rec, ensure_ascii=False) + "\n"); f.flush()
                counts[rec["branch"]] = counts.get(rec["branch"], 0) + 1
                sig = f"s0={rec['s0']:.2f} s1={rec['s1']:.2f}" if "s1" in rec else " " * 15
                print(f"[{ds} {i + 1}/{hi}] P(declined)={pdec:.2f} {sig} {rec['branch']:15s} {time.time() - ts:5.1f}s",
                      flush=True)
        print(f"  -> {o}  branches {counts}", flush=True)


if __name__ == "__main__":
    main()
