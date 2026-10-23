#!/bin/bash
# Run FwS's own eval scripts (eval_math.py on their 1024-problem MATH L3-5 test split;
# eval_ood_math.py on AMC, GSM8K and MATH-TTT, i.e. MATH500) for one model, one shard
# per visible GPU. Settings are theirs: chat template, temperature 0.6, 1856 new tokens,
# one sample per problem.
#   bash scripts/fws/eval_fws.sh <fws_dir> <model_path_or_id> <out_dir>
set -euo pipefail

root=${1:?usage: eval_fws.sh <fws_dir> <model> <out_dir>}
model=${2:?}
out=${3:?}
repo="$root/finetuning-with-sampling"
python="$root/eval-venv/bin/python"
num_gpus=$(nvidia-smi -L | wc -l)
if [[ -e "$out/math" || -e "$out/ood" ]]; then
    echo "$out already holds results; their scripts append, so use a fresh directory" >&2
    exit 1
fi
mkdir -p "$out"
out=$(cd "$out" && pwd)

export VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_USE_FLASHINFER_SAMPLER=0
export PYTHONPATH="$repo:$repo/grader_utils"
cd "$repo"
for ((gpu = 0; gpu < num_gpus; gpu++)); do
    (
        export CUDA_VISIBLE_DEVICES=$gpu
        "$python" eval_math.py --model_path="$model" --out_name=model --save_str="$out/math" \
            --batch_idx=$gpu --num_batches=$num_gpus > "$out/math_$gpu.log" 2>&1
        for dataset in AMC MATH-TTT GSM8K; do
            "$python" eval_ood_math.py --model_str="$model" --dataset=$dataset --save_str="$out/ood" \
                --batch_idx=$gpu --num_batches=$num_gpus --seed=0 > "$out/${dataset}_$gpu.log" 2>&1
        done
    ) &
done
wait
"$python" - "$out" <<'PY'
import glob, json, sys
out = sys.argv[1]
def acc(pattern):
    rows = [json.loads(line) for path in glob.glob(f"{out}/{pattern}") for line in open(path)]
    return len(rows), sum(r["is_correct"] for r in rows) / max(len(rows), 1)
summary = {name: dict(zip(("n", "accuracy"), acc(pattern))) for name, pattern in [
    ("MATH(3,4,5)", "math/eval_model_*.jsonl"), ("AMC", "ood/*_AMC_*.jsonl"),
    ("MATH500", "ood/*_MATH-TTT_*.jsonl"), ("GSM8K", "ood/*_GSM8K_*.jsonl")]}
json.dump(summary, open(f"{out}/summary.json", "w"), indent=2)
print(json.dumps(summary, indent=2))
PY
