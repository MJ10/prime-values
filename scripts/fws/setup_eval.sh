#!/bin/bash
# Prepare the Finetuning-with-Sampling (FwS) evaluation harness in its own directory and venv,
# isolated from prime-values: their repo pinned to a commit, plus the two pieces the release
# omits.
#   bash scripts/fws/setup_eval.sh "$SCRATCH/fws"
set -euo pipefail

root=${1:?usage: setup_eval.sh <dir>}
FWS_COMMIT=e74ab5eea6448bf78a4cff57179bbab8687cd243
PRM800K_COMMIT=7ecc794703b2877f63226f2477a49b34f9b25163

mkdir -p "$root"
cd "$root"
if [[ ! -d finetuning-with-sampling ]]; then
    git clone -q https://github.com/aakaran/finetuning-with-sampling.git
fi
git -C finetuning-with-sampling checkout -q "$FWS_COMMIT"
repo="$root/finetuning-with-sampling"

# Their math_grader.py is OpenAI's prm800k grader.py with one import changed, and imports
# grader_utils/math_normalize.py, which the release does not include.
curl -sSfL -o "$repo/grader_utils/math_normalize.py" \
    "https://raw.githubusercontent.com/openai/prm800k/$PRM800K_COMMIT/prm800k/grading/math_normalize.py"

# eval_math.py and eval_ood_math.py call format_prompt (for a logged likelihood, not for
# grading) but only boost_math.py defines it. Their `from constants import *` picks up this
# verbatim copy.
if ! grep -q "def format_prompt" "$repo/constants.py"; then
    cat >> "$repo/constants.py" <<'PY'


def format_prompt(text, model_type, tokenizer):
    if model_type == "base":
        format_str = text
    elif model_type == "chat":
        answer_context = [{"role": "user", "content": text}]
        format_str = tokenizer.apply_chat_template(answer_context, tokenize=False, add_generation_prompt=True)
    return format_str
PY
fi

# Versions from their environment.yml.
if [[ ! -x eval-venv/bin/python ]]; then
    uv venv -q --python 3.12 eval-venv
fi
uv pip install -q --python eval-venv/bin/python \
    vllm==0.11.2 transformers==4.57.6 pandas==3.0.1 pyarrow==23.0.1 sympy==1.14.0 pylatexenc==2.10 \
    tqdm==4.67.3 math-verify==0.9.0
echo "FwS eval harness ready in $root"
