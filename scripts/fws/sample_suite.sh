#!/bin/bash
# Sampler diagnostics for one model on FwS's MATH(3,4,5) test split, on 2 GPUs:
#   t06: 64 samples per problem at FwS's eval temperature 0.6 (pass@k, mean accuracy)
#   t10: 32 samples per problem at temperature 1 with sequence logprobs, scored under the
#        base model (reverse KL to p_C and log Z estimates)
#   base_correct: the base model's correct temperature-1 samples (exact samples of p_C),
#        scored under this model (forward KL from p_C); needs the base model's run
#   bash scripts/fws/sample_suite.sh <fws_dir> <model> <out_dir> [<base_out_dir>]
set -euo pipefail

root=${1:?usage: sample_suite.sh <fws_dir> <model> <out_dir> [<base_out_dir>]}
model=${2:?}
out=${3:?}
base_out=${4:-}
base_model=${FWS_BASE_MODEL:-Qwen/Qwen2.5-3B}
repo="$root/finetuning-with-sampling"
python=${FWS_PYTHON:-$root/eval-venv/bin/python}
tool="$(cd "$(dirname "$0")" && pwd)/sample_eval.py"
export VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_USE_FLASHINFER_SAMPLER=0
mkdir -p "$out"

CUDA_VISIBLE_DEVICES=0 "$python" "$tool" sample --fws "$repo" --model "$model" --temperature 0.6 --n 64 \
    --out "$out/t06.jsonl" > "$out/sample_t06.log" 2>&1 &
pids=($!)
CUDA_VISIBLE_DEVICES=1 "$python" "$tool" sample --fws "$repo" --model "$model" --temperature 1.0 --n 32 \
    --out "$out/t10.jsonl" > "$out/sample_t10.log" 2>&1 &
pids+=($!)
for pid in "${pids[@]}"; do wait "$pid"; done

for name in t06 t10; do
    "$python" "$tool" grade --fws "$repo" --samples "$out/$name.jsonl" --out "$out/$name.graded.jsonl" \
        > "$out/grade_$name.log" 2>&1
    rm "$out/$name.jsonl"
done

CUDA_VISIBLE_DEVICES=0 "$python" "$tool" score --fws "$repo" --model "$base_model" \
    --samples "$out/t10.graded.jsonl" --out "$out/t10.base_logprob.jsonl" > "$out/score_t10.log" 2>&1 &
pids=($!)
if [[ -n "$base_out" ]]; then
    CUDA_VISIBLE_DEVICES=1 "$python" "$tool" score --fws "$repo" --model "$model" --only-correct \
        --samples "$base_out/t10.graded.jsonl" --out "$out/base_correct.model_logprob.jsonl" \
        > "$out/score_base_correct.log" 2>&1 &
    pids+=($!)
fi
for pid in "${pids[@]}"; do wait "$pid"; done
echo "sample suite done: $out"
