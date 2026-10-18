# /// script
# requires-python = ">=3.12"
# dependencies = ["matplotlib", "numpy"]
# ///
"""Summarize random-number study runs (configs/random_number) from their saved
train rollouts: validity, KL to uniform, and Figure 2-style histograms.

    uv run scripts/random_number_report.py base=~/scratch/prime-values/mila_1 \\
        grpo=~/scratch/prime-values/mila_2 ... --out outputs/random_number_report
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

LOW, HIGH = 0, 100
SUPPORT = HIGH - LOW + 1


def load_numbers(run_dir: Path) -> dict[int, list[int]]:
    """Per step, the ``number`` metric of every train rollout (-1 when invalid)."""
    steps = {}
    for path in run_dir.glob("*/rollouts/step_*/train_rollouts.jsonl"):
        step = int(path.parent.name.removeprefix("step_"))
        with path.open() as f:
            steps[step] = [int(json.loads(line)["metrics"]["number"]) for line in f]
    if not steps:
        raise FileNotFoundError(f"no train rollouts under {run_dir}/*/rollouts")
    return steps


def summarize(numbers: list[int]) -> dict[str, float]:
    """Validity, plus KL and entropy of the valid numbers' empirical distribution
    against the uniform target on ``[LOW, HIGH]`` (the paper's KL)."""
    values = np.asarray(numbers)
    valid = values[values >= 0]
    counts = np.bincount(valid - LOW, minlength=SUPPORT)
    probs = counts[counts > 0] / max(len(valid), 1)
    return {
        "samples": len(values),
        "valid": len(valid) / len(values),
        "kl_to_uniform": float(np.sum(probs * np.log(probs * SUPPORT))),
        "entropy_bits": float(-np.sum(probs * np.log2(probs))),
        "unique": int((counts > 0).sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", help="label=run_dir pairs, in plotting order")
    parser.add_argument("--window", type=int, default=10, help="final steps pooled into the histogram and table")
    parser.add_argument("--out", type=Path, default=Path("outputs/random_number_report"))
    args = parser.parse_args()

    runs = {}
    for spec in args.runs:
        label, _, path = spec.partition("=")
        runs[label] = load_numbers(Path(path).expanduser())
    args.out.mkdir(parents=True, exist_ok=True)

    finals = {}
    for label, steps in runs.items():
        final_steps = sorted(steps)[-args.window :]
        finals[label] = [n for step in final_steps for n in steps[step]]

    rows = [f"| arm | samples | valid | KL to uniform | entropy (bits, max {np.log2(SUPPORT):.2f}) | unique |"]
    rows.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    for label, numbers in finals.items():
        s = summarize(numbers)
        rows.append(
            f"| {label} | {s['samples']} | {s['valid']:.1%} | {s['kl_to_uniform']:.4f} "
            f"| {s['entropy_bits']:.2f} | {s['unique']}/{SUPPORT} |"
        )
    table = "\n".join(rows)
    print(f"Last {args.window} steps per run:\n\n{table}")
    (args.out / "summary.md").write_text(table + "\n")

    fig, axes = plt.subplots(1, len(finals), figsize=(4.5 * len(finals), 3.2), sharey=True, squeeze=False)
    for ax, (label, numbers) in zip(axes[0], finals.items(), strict=True):
        values = np.asarray(numbers)
        valid = values[values >= 0]
        ax.bar(np.arange(LOW, HIGH + 1), np.bincount(valid - LOW, minlength=SUPPORT) / len(values), width=1.0)
        ax.axhline(1 / SUPPORT, color="black", linestyle="--", linewidth=0.8)
        ax.set_yscale("log")
        ax.set_title(f"{label} ({len(valid) / len(values):.1%} valid)")
        ax.set_xlabel("number")
    axes[0][0].set_ylabel("fraction of samples")
    fig.tight_layout()
    fig.savefig(args.out / "histograms.png", dpi=150)

    fig, (ax_valid, ax_kl) = plt.subplots(1, 2, figsize=(10, 3.2))
    for label, steps in runs.items():
        xs = sorted(steps)
        stats = [summarize(steps[x]) for x in xs]
        ax_valid.plot(xs, [s["valid"] for s in stats], label=label)
        ax_kl.plot(xs, [s["kl_to_uniform"] for s in stats], label=label)
    ax_valid.set(xlabel="step", ylabel="valid fraction")
    ax_kl.set(xlabel="step", ylabel="KL(empirical || uniform), per step", yscale="log")
    ax_valid.legend()
    fig.tight_layout()
    fig.savefig(args.out / "curves.png", dpi=150)
    print(f"Wrote {args.out}/summary.md, histograms.png, curves.png")


if __name__ == "__main__":
    main()
