import re

from datasets import Dataset

import verifiers as vf

PROMPT = "The following is a random integer drawn uniformly between {low} and {high}: "
LEADING_DIGITS = re.compile(r"[0-9]+")


def parse_number(text: str, num_tokens: int, low: int, high: int) -> int | None:
    """The integer a completion names, or None unless it is a canonical integer
    in ``[low, high]`` followed by exactly one terminating token.

    Assumes one token per digit (true of Qwen tokenizers), so every number has
    the same set of valid completions: its digits, then any token that ends
    generation (EOS or one matching a sampling stop string)."""
    match = LEADING_DIGITS.match(text)
    if match is None:
        return None
    digits = match.group()
    if num_tokens != len(digits) + 1 or digits != str(int(digits)):
        return None
    number = int(digits)
    return number if low <= number <= high else None


def load_environment(low: int = 0, high: int = 100, num_examples: int = 1024) -> vf.Environment:
    """Random integer generation (Hu et al., 2023, arXiv:2310.04363, Section 2).

    Every example is the same completion-style prompt. Sample with stop strings
    covering separators (newline, space, punctuation) so a rollout can end
    right after any digit, like the paper's terminate-at-every-state GFlowNet.
    Valid rollouts (see ``parse_number``) get reward 1 and invalid ones 0; the
    ``number`` metric records the parsed integer (-1 when invalid)."""
    prompt = PROMPT.format(low=low, high=high)

    def completion_number(completion, state) -> int | None:
        step = state["trajectory"][-1]
        if step["is_truncated"]:
            return None
        return parse_number(completion[-1]["content"] or "", len(step["tokens"]["completion_ids"]), low, high)

    def valid_number(completion, state, **kwargs) -> float:
        return float(completion_number(completion, state) is not None)

    def number(completion, state, **kwargs) -> float:
        parsed = completion_number(completion, state)
        return -1.0 if parsed is None else float(parsed)

    rubric = vf.Rubric(funcs=[valid_number, number], weights=[1.0, 0.0])
    dataset = Dataset.from_dict({"question": [prompt] * num_examples, "answer": [""] * num_examples})
    return vf.SingleTurnEnv(dataset=dataset, rubric=rubric)
