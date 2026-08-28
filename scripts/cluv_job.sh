#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1

set -euo pipefail

if (( $# == 0 )); then
    echo "Usage: cluv submit <cluster> -- <prime-rl command> [args...]" >&2
    exit 2
fi

output_pattern="${SBATCH_OUTPUT:?This job script must be submitted through cluv}"
run_dir="${output_pattern%/*}"
run_dir="${run_dir//%j/${SLURM_JOB_ID}}"
run_dir="${run_dir//%A/${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}}"
run_dir="${run_dir//%a/${SLURM_ARRAY_TASK_ID:-0}}"
mkdir -p "$run_dir"

command=("$@")
case "${command[0]}" in
    rl|sft|inference|trainer|orchestrator|value-trainer|value-evaluator)
        has_output_dir=false
        for arg in "${command[@]}"; do
            if [[ "$arg" == "--output-dir" || "$arg" == --output-dir=* ]]; then
                has_output_dir=true
                break
            fi
        done
        if [[ "$has_output_dir" == false ]]; then
            command+=(--output-dir "$run_dir")
        fi
        ;;
esac

export PRIME_RL_CLUV_OUTPUT_DIR="$run_dir"
echo "Output directory: $run_dir"
printf 'Running command:'
printf ' %q' "${command[@]}"
printf '\n'

srun --ntasks=1 uv run --no-sync "${command[@]}"
