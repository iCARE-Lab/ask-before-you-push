# Ask Before You Push

**Training-free, self-checked decoding for reducing LLM over-refusal.**

Safety-aligned LLMs often refuse harmless requests that merely *sound* sensitive. Out of the box, Llama-3-8B-Instruct refuses 69.7% and Gemma-2-9B-it 76.2% of the harmless prompts in OR-Bench-Hard. Ask Before You Push lets the model decide for itself whether a refusal is unnecessary. Only then does it steer the first few tokens toward answering, and it lets the model check the result for harm. It needs no training, no extra model and no per-model tuning.

## Method

For every prompt, using one model and the same constants for all models:

1. **Answer first.** Greedy decoding gives the plain answer. If the model, asked about its own answer, says it answered the request, that answer is returned.
2. **Refusal self-diagnosis.**
   - *s0* is the model's probability of opening its reply with a refusal (e.g. "I cannot").
   - *s1* is the same probability under a system prompt stating that the request is benign.
3. **Direction check.** If *s1 ≥ s0*, reassurance does not lower the refusal, so the refusal is kept.
4. **Depth check.** If *s1 ≥ 0.5*, the model still prefers refusing even when told the request is fine, so the refusal is kept.
5. **Adaptive push.** The first *m = 5* tokens are decoded from

   z = z_base + β (z_perm − z_base),  β = 3 · σ(20 · (0.5 − s1)).

   Here z_base and z_perm are the logits without and with the benign system prompt. The remaining tokens use plain greedy decoding.
6. **Output self-check.** If the model judges its pushed answer harmful (P > 0.5), the original answer is returned.

The code is in `abyp/generate.py`, and every fixed prompt is in `abyp/prompts.py`.

## Setup

```bash
pip install -r requirements.txt
python data/fetch_data.py          # six public benchmarks -> data/*.jsonl
```

Llama-3 and Gemma-2 are gated on Hugging Face. Accept their licences and run `huggingface-cli login` first.

## Usage

```bash
# Generate (steps 1-6). One GPU per run; resumable.
python -m abyp.generate --model meta-llama/Meta-Llama-3-8B-Instruct --out outputs/llama

# Judge every answer of ours and of the plain model with an LLM judge. Needs OPENAI_API_KEY; cached.
python eval/judge.py --run outputs/llama

# Table: refusal rate (benign sets), attack success rate (harmful sets), exact McNemar p-values vs ours.
python eval/report.py --run outputs/llama
```

`scripts/run_all.sh` runs everything for the three models.

To compare with another method, such as [AdaCD](https://arxiv.org/abs/2604.17132), produce its outputs with that method's official code on the same `data/*.jsonl`. Cap them at 512 new tokens, like ours. Save one `{"id", "response"}` per line in `<dir>/<dataset>.jsonl`, then add `--method adacd=<dir>` to both `judge.py` and `report.py`.

Each output record keeps the plain answer (`base`), the method's answer (`response`), the branch taken, and *s0*, *s1*, β and the self-check probabilities. Use these to see why a prompt was or was not pushed.

## Results

Judge: GPT-6.1-terra. Each row lists the percentage on every prompt of each benchmark; lower is better.
† / ‡: significantly lower / higher than AdaCD (paired exact McNemar test, p < 0.05).

| Model | Method | OR-Bench-Hard ↓ | OKTest ↓ | XSTest-Safe ↓ | XSTest-Unsafe ↓ | AdvBench ↓ | JailBench ↓ |
|---|---|---|---|---|---|---|---|
| Llama-3-8B | Base | 69.7 | 8.7 | 3.2 | 0.0 | 0.8 | 0.0 |
| | AdaCD | 34.3 | 4.0 | 0.8 | 0.0 | 2.5 | 3.0 |
| | **Ours** | **29.3**† | 4.3 | 1.6 | 1.0 | 0.8† | 0.0 |
| Gemma-2-9B | Base | 76.2 | 17.0 | 14.8 | 0.0 | 0.4 | 0.0 |
| | AdaCD | 65.2 | 13.3 | 10.8 | 0.0 | 0.4 | 0.0 |
| | **Ours** | **30.6**† | **9.3**† | **5.2**† | 0.0 | 0.6 | 2.0 |
| Qwen2.5-7B | Base | 14.6 | 6.3 | 1.2 | 0.0 | 0.4 | 2.0 |
| | AdaCD | 9.3 | 4.7 | 0.8 | 0.5 | 1.3 | 6.0 |
| | **Ours** | 13.3‡ | 3.3 | 2.4 | 0.0 | 0.4 | 2.0 |

The benign sets (OR-Bench-Hard, OKTest, XSTest-Safe) report the refusal rate. The harmful sets (XSTest-Unsafe, AdvBench, JailBench) report the attack success rate.

## Responsible use

The harmful benchmarks are downloaded from their original sources and are not redistributed here. Model outputs on them are not published. The method changes decoding only when the model itself judges a refusal unnecessary, and it keeps an output safety check. It has not been tested against adaptive attacks. Do not treat it as a safety guarantee.

## Citation

```bibtex
@misc{ming2026askbeforeyoupush,
  title  = {Ask Before You Push: Training-Free, Self-Checked Decoding for Reducing LLM Over-Refusal},
  author = {Ming, Nong and Imran, Muhammad},
  year   = {2026},
  note   = {iCARE Lab, Kennesaw State University},
  url    = {https://github.com/iCARE-Lab/ask-before-you-push}
}
```

## License

MIT
