"""Build SFT datasets for the FwS SFT baselines, one ``<out_dir>/<name>/train.parquet``
(prompt/completion columns) per dataset, loadable as ``[data] name = "<out_dir>/<name>"``.

- boosted_released: all 8229 released boosted traces, as FwS trained on them (268 of their
  problems are MATH500 test problems, absent from train.parquet)
- boosted_clean: the 7961 boosted traces whose problem is in train.parquet
- expert: the 7962 expert solutions of train.parquet (FwS's "SFT" row)

  uv run python scripts/fws/sft_data.py <fws_repo> <out_dir>
"""

import json
import sys
from pathlib import Path

import pandas as pd
from fws_math import build_question


def main() -> None:
    repo, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    train = pd.read_parquet(repo / "math_data/train.parquet")
    train_problems = set(train["problem"])
    traces = [json.loads(line) for line in open(repo / "math_data/sampling_traces_math_qwen.jsonl")]
    datasets = {
        "boosted_released": [(t["problem"], t["trace"]) for t in traces],
        "boosted_clean": [(t["problem"], t["trace"]) for t in traces if t["problem"] in train_problems],
        "expert": list(zip(train["problem"], train["solution"])),
    }
    for name, pairs in datasets.items():
        frame = pd.DataFrame({"prompt": [build_question(p) for p, _ in pairs], "completion": [c for _, c in pairs]})
        (out / name).mkdir(exist_ok=True)
        frame.to_parquet(out / name / "train.parquet")
        print(f"{name}: {len(frame)} examples")


if __name__ == "__main__":
    main()
