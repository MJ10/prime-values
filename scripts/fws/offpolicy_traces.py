"""Write FwS's released boosted traces as gfn off-policy completions.

Keeps the traces whose problem is in FwS's train.parquet (7961 of 8229; the other 268
are MATH500 problems) and labels each with the fws-math env's reward: math-verify on
the trace's last boxed answer against the expert solution's.

  uv run python scripts/fws/offpolicy_traces.py <fws_repo> <out.jsonl>
"""

import json
import sys
from pathlib import Path

import pandas as pd
from fws_math import build_question, last_boxed
from verifiers.rubrics.math_rubric import verify_response
from verifiers.utils.data_utils import extract_boxed_answer


def main() -> None:
    repo, out = Path(sys.argv[1]), Path(sys.argv[2])
    train = pd.read_parquet(repo / "math_data/train.parquet")
    gold = {problem: last_boxed(solution) for problem, solution in zip(train["problem"], train["solution"])}
    traces = [json.loads(line) for line in open(repo / "math_data/sampling_traces_math_qwen.jsonl")]
    kept = [t for t in traces if t["problem"] in gold]
    with open(out, "w") as f:
        for t in kept:
            reward, _ = verify_response(extract_boxed_answer(t["trace"], strict=True), gold[t["problem"]], 50_000)
            f.write(
                json.dumps({"prompt": build_question(t["problem"]), "completion": t["trace"], "reward": reward}) + "\n"
            )
    rewards = [json.loads(line)["reward"] for line in open(out)]
    fws_correct = sum(int(t["is_correct"]) for t in kept) / len(kept)
    print(
        f"kept {len(kept)}/{len(traces)} traces; math-verify accuracy {sum(rewards) / len(rewards):.4f}, FwS label {fws_correct:.4f}"
    )


if __name__ == "__main__":
    main()
