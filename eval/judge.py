"""LLM judge for the outputs of abyp.generate (and, optionally, of other methods on the same prompts).

Every response gets the rubric in rubric.py: compliance (XSTest scheme) and harmful (content-based).
The judge sees only the request and one response, never the method name. Results are cached in
<run>/judgments_<judge>.jsonl, keyed by a hash of (request, response), so reruns only pay for new responses.

  export OPENAI_API_KEY=...
  python eval/judge.py --run outputs/llama                                  # ours + the plain model
  python eval/judge.py --run outputs/llama --method adacd=adacd_out/llama   # also another method's outputs
Another method's folder holds <dataset>.jsonl files with {"id", "response"}, ids matching data/<dataset>.jsonl.
"""
import argparse, glob, hashlib, json, os, sys, time
from concurrent.futures import ThreadPoolExecutor

import openai
from openai import OpenAI

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rubric import SYSTEM, USER


def key(prompt, response):
    return hashlib.sha1(f"{prompt}\x00{response}".encode()).hexdigest()[:20]


def load_responses(run, methods):
    """{method: {(dataset, id): response}} plus {(dataset, id): prompt}; methods "ours" and "base" come from the run."""
    resp, prompts = {"ours": {}, "base": {}}, {}
    for p in sorted(glob.glob(f"{run}/*.jsonl")):
        if os.path.basename(p).startswith("judgments_"): continue
        for r in map(json.loads, open(p)):
            k = (r["dataset"], r["id"])
            prompts[k] = r["user_text"]; resp["ours"][k] = r["response"]; resp["base"][k] = r["base"]
    for name, folder in methods:
        resp[name] = {}
        for p in sorted(glob.glob(f"{folder}/*.jsonl")):
            ds = os.path.basename(p)[:-len(".jsonl")]
            for r in map(json.loads, open(p)):
                if (ds, r["id"]) in prompts: resp[name][(ds, r["id"])] = r["response"]
    return resp, prompts


def judge_one(client, judge, it):
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": USER.format(prompt=it["prompt"], response=it["response"] or "(empty response)")}]
    out = {"key": it["key"], "judge": judge}
    for attempt in range(4):
        try:
            r = client.chat.completions.create(model=judge, messages=msgs, response_format={"type": "json_object"})
            m = r.choices[0].message
            if getattr(m, "refusal", None):
                out["status"] = "judge_refused"; return out
            try:
                out.update(json.loads(m.content)); out["status"] = "ok"
            except Exception:
                out["status"] = "bad_json"
            return out
        except openai.BadRequestError:                 # content-policy blocks land here
            out["status"] = "blocked"; return out
        except (openai.RateLimitError, openai.APIConnectionError, openai.InternalServerError):
            time.sleep(2 ** attempt * 5)
    out["status"] = "error"; return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="output folder of abyp.generate")
    ap.add_argument("--method", action="append", default=[], metavar="NAME=DIR", help="another method's outputs")
    ap.add_argument("--judge", default="gpt-6.1-terra")
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()

    methods = [m.split("=", 1) for m in a.method]
    resp, prompts = load_responses(a.run, methods)
    cache = f"{a.run}/judgments_{a.judge}.jsonl"
    done = {r["key"] for r in map(json.loads, open(cache)) if r["status"] != "error"} if os.path.exists(cache) else set()
    todo, seen = [], set()
    for rs in resp.values():
        for k, response in rs.items():
            kk = key(prompts[k], response)
            if kk in done or kk in seen: continue
            seen.add(kk); todo.append({"key": kk, "prompt": prompts[k], "response": response})
    print(f"{len(todo)} responses to judge with {a.judge} ({len(done)} cached)", flush=True)

    client = OpenAI()
    with open(cache, "a") as f, ThreadPoolExecutor(a.workers) as ex:
        for n, r in enumerate(ex.map(lambda it: judge_one(client, a.judge, it), todo), 1):
            f.write(json.dumps(r, ensure_ascii=False) + "\n"); f.flush()
            if n % 500 == 0: print(f"  {n}/{len(todo)}", flush=True)
    print("saved", cache)


if __name__ == "__main__":
    main()
