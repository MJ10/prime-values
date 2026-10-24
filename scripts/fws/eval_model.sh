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

bash "$here/eval_fws.sh" "$root" "$model" "$out/fws_eval"
bash "$here/sample_suite.sh" "$root" "$model" "$out/suite" "$base_suite"
"$root/eval-venv/bin/python" "$here/report.py" "$out/suite" "$base_suite" --json "$out/suite/report.json"
