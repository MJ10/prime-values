#!/bin/bash
# Full evaluation of one checkpoint on 2 GPUs: FwS's own eval scripts (eval_fws.sh), then
# the sampler diagnostics against the base model's suite (sample_suite.sh).
#   bash scripts/fws/eval_model.sh <fws_dir> <model> <out_dir> <base_suite_dir>
set -euo pipefail

root=${1:?usage: eval_model.sh <fws_dir> <model> <out_dir> <base_suite_dir>}
model=${2:?}
out=${3:?}
base_suite=${4:?}
here="$(cd "$(dirname "$0")" && pwd)"

# With FWS_LOCAL_VENV=1, run from a node-local copy of the eval venv. Mila's scratch
# filesystem intermittently fails file reads when a process starts, and this job starts
# eight; the copy is retried because the same failure can hit it.
if [[ "${FWS_LOCAL_VENV:-0}" == 1 ]]; then
    for attempt in 1 2 3; do
        if rsync -a "$root/eval-venv/" "${SLURM_TMPDIR:?FWS_LOCAL_VENV needs SLURM_TMPDIR}/eval-venv/"; then
            break
        fi
        echo "eval-venv copy attempt $attempt failed" >&2
        if [[ $attempt == 3 ]]; then
            exit 1
        fi
    done
    export FWS_PYTHON="$SLURM_TMPDIR/eval-venv/bin/python"
fi
python=${FWS_PYTHON:-$root/eval-venv/bin/python}

# Checkpoints are saved by transformers 5, whose tokenizer_config.json the eval venv's
# transformers 4.57 cannot read. Training never changes the tokenizer, so a checkpoint is
# evaluated as its weights and config plus the base model's tokenizer and generation config
# (what FwS's verl SFT saves alongside its checkpoints).
if [[ -d "$model" ]]; then
    base_dir=$("$python" -c \
        "from huggingface_hub import snapshot_download; print(snapshot_download('Qwen/Qwen2.5-3B', local_files_only=True))")
    eval_dir="$out/model"
    mkdir -p "$eval_dir"
    for file in "$model"/*.safetensors "$model"/model.safetensors.index.json "$model"/config.json; do
        if [[ -e "$file" ]]; then
            ln -sfn "$file" "$eval_dir/"
        fi
    done
    for file in tokenizer.json tokenizer_config.json vocab.json merges.txt generation_config.json; do
        cp -L "$base_dir/$file" "$eval_dir/"
    done
    model="$eval_dir"
fi

bash "$here/eval_fws.sh" "$root" "$model" "$out/fws_eval"
bash "$here/sample_suite.sh" "$root" "$model" "$out/suite" "$base_suite"
"$python" "$here/report.py" "$out/suite" "$base_suite" --json "$out/suite/report.json"
