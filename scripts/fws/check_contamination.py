"""Check FwS's released training data against their evaluation sets.

Compares problem texts in math_data/sampling_traces_math_qwen.jsonl (the boosted traces
used for Sampling SFT) and math_data/train.parquet with ood_data/MATH-TTT.json (MATH500)
and the 1024-problem test split, by exact match and after whitespace normalization.

  python scripts/fws/check_contamination.py <fws_repo> [--examples N]
"""

import argparse
import json
import re
from pathlib import Path

import pandas as pd


def last_boxed(text: str) -> str | None:
    """Contents of the last ``\\boxed{...}``, as FwS's ``parse_answer`` extracts it."""
    start = text.rfind("\\boxed{")
    if start < 0:
        return None
    depth = 0
    for end in range(start + len("\\boxed"), len(text)):
        depth += {"{": 1, "}": -1}.get(text[end], 0)
        if depth == 0:
            return text[start + len("\\boxed{") : end]
    return None


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def test_question(wrapped: str) -> str:
    """The problem inside a test-split prompt, extracted as eval_math.py does."""
    start = wrapped.index("\nUser: ") + len("\nUser: ")
    return wrapped[start : wrapped.rfind("Show your work")]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("repo", type=Path)
    parser.add_argument("--examples", type=int, default=5)
    args = parser.parse_args()

    traces = [json.loads(line) for line in open(args.repo / "math_data/sampling_traces_math_qwen.jsonl")]
    train = pd.read_parquet(args.repo / "math_data/train.parquet")
    test = pd.read_parquet(args.repo / "math_data/test.parquet")
    math500 = json.load(open(args.repo / "ood_data/MATH-TTT.json"))

    m500 = {normalize(d["prompt"]): d for d in math500}
    train_problems = {normalize(p) for p in train["problem"]}
    test_problems = {normalize(test_question(q)) for q in test["question"]}

    in_train = [t for t in traces if normalize(t["problem"]) in train_problems]
    extra = [t for t in traces if normalize(t["problem"]) not in train_problems]
    extra_in_m500 = [t for t in extra if normalize(t["problem"]) in m500]
    exact_in_m500 = [t for t in extra if t["problem"] in {d["prompt"] for d in math500}]

    print(f"boosted traces: {len(traces)} ({len({normalize(t['problem']) for t in traces})} distinct problems)")
    print(f"  problem in train.parquet ({len(train)} problems): {len(in_train)}")
    print(f"  problem not in train.parquet: {len(extra)}")
    print(
        f"    of those, in MATH500 (whitespace-normalized): {len(extra_in_m500)}; exact string match: {len(exact_in_m500)}"
    )
    print(
        f"  traces whose problem is a MATH500 problem: {sum(normalize(t['problem']) in m500 for t in traces)} of 500 MATH500 problems"
    )
    print(f"train.parquet ∩ MATH500: {len(train_problems & set(m500))}")
    print(f"test split ({len(test)}) ∩ MATH500: {len(test_problems & set(m500))}")
    print(f"boosted-trace problems ∩ test split: {len({normalize(t['problem']) for t in traces} & test_problems)}")

    # Do the contaminated traces carry MATH500's gold answer?
    labelled_correct = sum(int(t["is_correct"]) for t in extra_in_m500)
    gold_in_trace = sum(
        normalize(last_boxed(t["trace"]) or "") == normalize(m500[normalize(t["problem"])]["answer"])
        for t in extra_in_m500
    )
    print(f"contaminated traces labelled correct by FwS: {labelled_correct}/{len(extra_in_m500)}")
    print(
        f"contaminated traces whose last \\boxed{{}} equals the MATH500 answer string: {gold_in_trace}/{len(extra_in_m500)}"
    )

    print("\nexamples (trace idx | MATH500 id | answer | trace's last boxed):")
    for t in extra_in_m500[: args.examples]:
        d = m500[normalize(t["problem"])]
        print(f"- idx {t['idx']} | {d['id']} | {d['answer']!r} | {last_boxed(t['trace'])!r}")
        print(f"  problem: {t['problem'][:110]!r}")


if __name__ == "__main__":
    main()
