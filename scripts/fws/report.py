"""Summarize scripts/fws/sample_suite.sh outputs: pass@k, mean accuracy, and distances to
FwS's target p_C(x) ∝ p_base(x | q) * 1[x correct] (floored at eps, as gfn trains).

Per problem q, with x_i ~ π at temperature 1 and d_i = log 1_eps[correct] + log p_base(x_i) - log π(x_i):
- log Z_q, the target's log-partition, from the base model's own temperature-1 samples:
  log(acc_q * (1 - eps) + eps) (base samples have d_i = log 1_eps exactly).
- ELBO_q = mean_i d_i <= log Z_q, so KL(π || p_C) = log Z_q - ELBO_q.
- Forward KL(p_C || π) = E_{x ~ p_C}[log p_base(x) - log π(x)] - log Z_q, using the base model's
  correct samples, which are exact samples of p_C.
Both KLs are averaged over problems the base model solved at least once (log Z_q is finite there);
problems it never solved are counted separately.

  python report.py <suite_dir> [<base_suite_dir>] [--eps 1e-4] [--json out.json]
"""

import argparse
import json
import math
import os
from math import comb

KS = (1, 2, 4, 8, 16, 32, 64)


def load(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k estimator (Chen et al., 2021)."""
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


def mean(values: list[float]) -> float:
    return math.fsum(values) / len(values) if values else float("nan")


def summarize(suite: str, base_suite: str | None, eps: float) -> dict:
    t06 = load(os.path.join(suite, "t06.graded.jsonl"))
    out: dict = {}
    counts = [(len(row["samples"]), sum(s["correct"] for s in row["samples"])) for row in t06]
    out["mean_acc_t0.6"] = mean([c / n for n, c in counts])
    out["pass@k_t0.6"] = {k: mean([pass_at_k(n, c, k) for n, c in counts]) for k in KS if k <= counts[0][0]}

    t10 = load(os.path.join(suite, "t10.graded.jsonl"))
    out["mean_acc_t1.0"] = mean([mean([s["correct"] for s in row["samples"]]) for row in t10])
    if base_suite is None:
        return out

    base_t10 = {row["idx"]: row for row in load(os.path.join(base_suite, "t10.graded.jsonl"))}
    base_logprob = {row["idx"]: row["logprob"] for row in load(os.path.join(suite, "t10.base_logprob.jsonl"))}
    log_z = {}
    for idx, row in base_t10.items():
        acc = mean([s["correct"] for s in row["samples"]])
        log_z[idx] = math.log(acc * (1 - eps) + eps)
    solved = {idx for idx, row in base_t10.items() if any(s["correct"] for s in row["samples"])}

    reverse_kl, elbo_gap_unsolved = [], []
    for row in t10:
        d = [
            math.log(1.0 if s["correct"] else eps) + prior - s["logprob"]
            for s, prior in zip(row["samples"], base_logprob[row["idx"]], strict=True)
        ]
        gap = log_z[row["idx"]] - mean(d)
        (reverse_kl if row["idx"] in solved else elbo_gap_unsolved).append(gap)
    out["reverse_kl_to_pC"] = mean(reverse_kl)
    out["reverse_kl_to_pC_unsolved_by_base"] = mean(elbo_gap_unsolved)

    forward_path = os.path.join(suite, "base_correct.model_logprob.jsonl")
    if os.path.exists(forward_path):
        model_logprob = {row["idx"]: row["logprob"] for row in load(forward_path)}
        forward_kl = []
        for idx in solved:
            pairs = [
                (s["logprob"], lp)
                for s, lp in zip(base_t10[idx]["samples"], model_logprob[idx], strict=True)
                if s["correct"]
            ]
            forward_kl.append(mean([base - model for base, model in pairs]) - log_z[idx])
        out["forward_kl_from_pC"] = mean(forward_kl)
    out["num_problems_solved_by_base_t1"] = len(solved)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("suite")
    parser.add_argument("base_suite", nargs="?")
    parser.add_argument("--eps", type=float, default=1e-4)
    parser.add_argument("--json")
    args = parser.parse_args()
    summary = summarize(args.suite, args.base_suite, args.eps)
    print(json.dumps(summary, indent=2))
    if args.json:
        with open(args.json, "w") as f:
            json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
