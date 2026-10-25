#!/bin/bash
# Run `rl` with a frozen prior server for gfn's base-model log-prior: serve the base
# model on GPU 0 (beside the policy's vLLM, which the configs hold to 0.55 of memory),
# wait until it is healthy, then launch rl. Submit through cluv:
#   uv run scripts/cluv_prime_rl.py submit fir -- bash scripts/fws/rl_with_prior.sh \
#     @ configs/fws_gfn/base.toml @ configs/fws_gfn/arms/gfn.toml
set -euo pipefail

model=${FWS_PRIOR_MODEL:-Qwen/Qwen2.5-3B}
port=${FWS_PRIOR_PORT:-8001}
# FWS_OUTPUT_DIR resumes an earlier run in place (a new job gets a new cluv directory).
out=${FWS_OUTPUT_DIR:-${PRIME_RL_CLUV_OUTPUT_DIR:?run through cluv, or set PRIME_RL_CLUV_OUTPUT_DIR}}
mkdir -p "$out/logs"

CUDA_VISIBLE_DEVICES=0 inference --model.name "$model" --server.port "$port" \
    --gpu-memory-utilization 0.3 > "$out/logs/prior.log" 2>&1 &
prior=$!
trap 'kill "$prior" 2>/dev/null || true' EXIT
until curl -sf "http://localhost:$port/health" > /dev/null; do
    kill -0 "$prior"  # exits (set -e) if the prior server died
    sleep 5
done
echo "Prior server $model ready on port $port"
rl "$@" --output-dir "$out"
