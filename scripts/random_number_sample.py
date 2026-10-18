"""Draw a large sample from one model under the random-number study's sampling
rule (configs/random_number/base.toml) and save the number counts, for the
paper's Figure 2 (which used 512k samples per model).

    uv run python scripts/random_number_sample.py <hf-model-or-weights-dir> \\
        --num-samples 512000 --out numbers.json

Pass the result to scripts/random_number_report.py as ``label=numbers.json``.
"""

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

from random_number import PROMPT, TERMINATOR, parse_number  # noqa: E402
from vllm import LLM, SamplingParams  # noqa: E402

LOW, HIGH = 0, 100
MAX_COMPLETION_TOKENS = 5
CHUNK = 65536


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model", help="Hugging Face model name or a run's weights/step_<n> directory")
    parser.add_argument("--num-samples", type=int, default=512_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path(os.environ.get("PRIME_RL_CLUV_OUTPUT_DIR", ".")) / "numbers.json")
    args = parser.parse_args()

    llm = LLM(model=args.model, seed=args.seed, enable_prefix_caching=True)
    tokenizer = llm.get_tokenizer()
    prompt_ids = tokenizer.encode(PROMPT.format(low=LOW, high=HIGH))
    # Every knob explicit, so nothing falls back to the model's generation_config.json.
    params = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        top_k=-1,
        min_p=0.0,
        repetition_penalty=1.0,
        max_tokens=MAX_COMPLETION_TOKENS,
        stop=[TERMINATOR],
        stop_token_ids=[tokenizer.eos_token_id],
    )

    counts = [0] * (HIGH - LOW + 1)
    invalid = 0
    for start in range(0, args.num_samples, CHUNK):
        size = min(CHUNK, args.num_samples - start)
        outputs = llm.generate([{"prompt_token_ids": prompt_ids}] * size, params, use_tqdm=False)
        for output in outputs:
            completion = output.outputs[0]
            ids = list(completion.token_ids)
            number = None
            if completion.finish_reason != "length":
                number = parse_number(tokenizer.decode(ids, skip_special_tokens=True), len(ids), LOW, HIGH)
            if number is None:
                invalid += 1
            else:
                counts[number - LOW] += 1
        print(f"{start + size}/{args.num_samples} samples, {invalid} invalid", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps({"model": args.model, "samples": args.num_samples, "invalid": invalid, "low": LOW, "counts": counts})
    )
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
