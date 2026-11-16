#!/bin/bash
# Full pipeline for the three models in the paper: download data, generate (steps 1-6), judge, report.
# Generation needs one GPU per model (bf16; ~20 GB for Gemma-2-9B). Judging needs OPENAI_API_KEY.
#   bash scripts/run_all.sh                 # all models, sequentially on the visible GPU
#   MODELS="llama" bash scripts/run_all.sh  # one model
set -e
cd "$(dirname "$0")/.."

declare -A HF=([llama]=meta-llama/Meta-Llama-3-8B-Instruct [gemma]=google/gemma-2-9b-it [qwen]=Qwen/Qwen2.5-7B-Instruct)
MODELS=${MODELS:-"llama gemma qwen"}

[ -f data/orbench.jsonl ] || python data/fetch_data.py

for m in $MODELS; do
  python -m abyp.generate --model "${HF[$m]}" --out "outputs/$m"
done

for m in $MODELS; do
  python eval/judge.py --run "outputs/$m"
  echo "== $m"; python eval/report.py --run "outputs/$m"
done
