#!/bin/bash
# Diagnostic: the base model's MATH(3,4,5) accuracy when the question is sent as plain
# text, without the chat template that FwS's eval applies (4 samples per problem at 0.6).
#   bash scripts/fws/raw_prompt_check.sh <fws_dir> <model> <out_dir>
set -euo pipefail

root=${1:?usage: raw_prompt_check.sh <fws_dir> <model> <out_dir>}
model=${2:?}
out=${3:?}
repo="$root/finetuning-with-sampling"
python=${FWS_PYTHON:-$root/eval-venv/bin/python}
tool="$(cd "$(dirname "$0")" && pwd)/sample_eval.py"
export VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_USE_FLASHINFER_SAMPLER=0
mkdir -p "$out"

"$python" "$tool" sample --fws "$repo" --model "$model" --temperature 0.6 --n 4 --raw-prompt --out "$out/t06.jsonl"
"$python" "$tool" grade --fws "$repo" --samples "$out/t06.jsonl" --out "$out/t06.graded.jsonl"
"$python" -c "
import json, sys
rows = [json.loads(line) for line in open(sys.argv[1])]
print('raw-prompt mean accuracy', sum(s['correct'] for r in rows for s in r['samples']) / sum(len(r['samples']) for r in rows))
" "$out/t06.graded.jsonl"
