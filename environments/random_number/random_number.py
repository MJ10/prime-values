import math
import re

from datasets import Dataset

import verifiers as vf

PROMPT = "The following is a random integer drawn uniformly between {low} and {high}: "
LEADING_DIGITS = re.compile(r"[0-9]+")
# The one stop action, like EOS in the paper's GPT-J setup (Qwen2.5-Instruct
# almost never emits EOS after this raw prompt, and ends most numbers with ".").
# It must also be the sampling stop string, so generation ends at the first
# token containing it.
TERMINATOR = "."


def parse_number(text: str, num_tokens: int, low: int, high: int) -> int | None:
    """The integer a completion names, or None unless it is a canonical integer
    in ``[low, high]`` followed by exactly one ``TERMINATOR`` token (merged
    tokens such as ``".\\n"`` or ``".\\n\\n"`` do not count).

    Assumes one token per digit (true of Qwen tokenizers), so every number has
    exactly one valid completion."""
    match = LEADING_DIGITS.match(text)
    if match is None:
        return None
    digits = match.group()
    if num_tokens != len(digits) + 1 or digits != str(int(digits)):
        return None
    if text[match.end() :] != TERMINATOR:
        return None
    number = int(digits)
    return number if low <= number <= high else None


def log_reward(valid: bool, num_tokens: int) -> float:
    """0 for a valid completion, and ``-8 - 8 * (1 + 2 + ... + L)`` for an invalid
    one of ``L`` tokens: the paper's per-position penalty, measured against its
    +8 for a valid number. The penalty grows quadratically with length, so the
    ~V^L invalid completions of each length carry about 2e-4 of the target mass
    in total."""
    return 0.0 if valid else -8.0 - 4.0 * num_tokens * (num_tokens + 1)


def load_environment(low: int = 0, high: int = 100, num_examples: int = 1024) -> vf.Environment:
    """Random integer generation (Hu et al., 2023, arXiv:2310.04363, Section 2).

    Every example is the same completion-style prompt. Sample with the stop
    string ``TERMINATOR`` so a rollout can end right after any digit, like the
    paper's terminate-at-every-state GFlowNet. The reward is ``exp(log_reward)``:
    1 for a valid rollout (see ``parse_number``) and at most ``e^-16`` otherwise,
    so it doubles as a validity reward for reward-maximizing arms and as the
    GFlowNet target. The ``valid_number`` metric is the 0/1 validity and
    ``number`` the parsed integer (-1 when invalid)."""
    prompt = PROMPT.format(low=low, high=high)

    def completion_number(completion, state) -> int | None:
        step = state["trajectory"][-1]
        if step["is_truncated"]:
            return None
        return parse_number(completion[-1]["content"] or "", len(step["tokens"]["completion_ids"]), low, high)

    def target_reward(completion, state, **kwargs) -> float:
        num_tokens = len(state["trajectory"][-1]["tokens"]["completion_ids"])
        return math.exp(log_reward(completion_number(completion, state) is not None, num_tokens))

    def valid_number(completion, state, **kwargs) -> float:
        return float(completion_number(completion, state) is not None)

    def number(completion, state, **kwargs) -> float:
        parsed = completion_number(completion, state)
        return -1.0 if parsed is None else float(parsed)

    rubric = vf.Rubric(funcs=[target_reward, valid_number, number], weights=[1.0, 0.0, 0.0])
    dataset = Dataset.from_dict({"question": [prompt] * num_examples, "answer": [""] * num_examples})
    return vf.SingleTurnEnv(dataset=dataset, rubric=rubric)
