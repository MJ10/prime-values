import os

import pandas as pd
import verifiers as vf
from datasets import Dataset

# Verbatim from the Finetuning-with-Sampling eval and boosting scripts
# (github.com/aakaran/finetuning-with-sampling, eval_math.py / boost_math.py).
# MATH_COT is never `.format`-ed there, so the model sees a literal "\boxed{{}}".
MATH_PROMPT = "You are an expert mathematician. Can you solve the following math problem? "
MATH_COT = "\n\nExplain your reasoning, thinking step by step, and return your final answer within \\boxed{{}}."


def build_question(problem: str) -> str:
    """The user message FwS sends for ``problem``; the chat template adds Qwen's default system prompt."""
    return MATH_PROMPT + problem + MATH_COT


def last_boxed(text: str) -> str | None:
    """Contents of the last ``\\boxed{...}`` (or ``\\fbox{...}``) in ``text``, matching
    FwS's ``parse_answer`` (``remove_boxed(last_boxed_only_string(...))``)."""
    start = text.rfind("\\boxed")
    if start < 0:
        start = text.rfind("\\fbox")
        if start < 0:
            return None
    depth = 0
    for end in range(start, len(text)):
        if text[end] == "{":
            depth += 1
        elif text[end] == "}":
            depth -= 1
            if depth == 0:
                boxed = text[start : end + 1]
                return boxed[len("\\boxed{") : -1] if boxed.startswith("\\boxed{") else None
    return None


def load_environment(data_path: str, math_verify_timeout: int = 5) -> vf.Environment:
    """MATH Levels 3-5 train split of Finetuning with Sampling (``math_data/train.parquet``
    in their repo: 7962 problems with expert solutions, disjoint from their 1024-problem
    test split and from MATH500).

    The prompt is FwS's, sent as a single user turn so Qwen's chat template supplies the
    same default system prompt as their ``apply_chat_template`` call. The gold answer is
    the last boxed expression of the expert solution, as in ``boost_math.py``; the 2
    solutions written as ``\\boxed 2`` (no braces), which FwS's parser also cannot read,
    are dropped, leaving 7960 problems. The reward is math-verify on the completion's
    last ``\\boxed{}`` (1 correct, 0 otherwise); FwS grades with a sympy-based checker
    instead, so the two can disagree on edge cases.
    Environment variables in ``data_path`` (e.g. ``$SCRATCH``) are expanded."""
    frame = pd.read_parquet(os.path.expandvars(data_path))
    pairs = [(problem, last_boxed(solution)) for problem, solution in zip(frame["problem"], frame["solution"])]
    pairs = [(problem, answer) for problem, answer in pairs if answer is not None]
    dataset = Dataset.from_dict({"question": [build_question(p) for p, _ in pairs], "answer": [a for _, a in pairs]})
    return vf.SingleTurnEnv(dataset=dataset, rubric=vf.MathRubric(timeout_seconds=math_verify_timeout))
