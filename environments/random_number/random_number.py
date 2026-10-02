import re

from datasets import Dataset

import verifiers as vf

PROMPT = "The following is a random integer drawn uniformly between {low} and {high}: "
CANONICAL_INTEGER = re.compile(r"(0|[1-9][0-9]*)\n?")


def parse_number(text: str, low: int, high: int) -> int | None:
    """The integer a completion names, or None unless it is exactly one canonical
    integer in ``[low, high]`` with at most a trailing newline terminator."""
    match = CANONICAL_INTEGER.fullmatch(text)
    if match is None:
        return None
    number = int(match.group(1))
    return number if low <= number <= high else None


def load_environment(low: int = 0, high: int = 100, num_examples: int = 1024) -> vf.Environment:
    """Random integer generation (Hu et al., 2023, arXiv:2310.04363, Section 2).

    Every example is the same completion-style prompt. A rollout is valid when
    it terminates (newline stop string or EOS, not the token limit) right after
    one canonical integer in ``[low, high]``. Valid rollouts get reward 1 and
    invalid ones 0; the ``number`` metric records the parsed integer (-1 when
    invalid) for distribution analysis."""
    prompt = PROMPT.format(low=low, high=high)
    parser = vf.Parser()

    def completion_number(completion, state) -> int | None:
        if state["is_truncated"]:
            return None
        return parse_number(parser.parse_answer(completion) or "", low, high)

    def valid_number(completion, state, **kwargs) -> float:
        return float(completion_number(completion, state) is not None)

    def number(completion, state, **kwargs) -> float:
        parsed = completion_number(completion, state)
        return -1.0 if parsed is None else float(parsed)

    rubric = vf.Rubric(funcs=[valid_number, number], weights=[1.0, 0.0], parser=parser)
    dataset = Dataset.from_dict({"question": [prompt] * num_examples, "answer": [""] * num_examples})
    return vf.SingleTurnEnv(dataset=dataset, parser=parser, rubric=rubric)
