"""Batched sampling, prefill scoring and FwS grading on the FwS MATH(3,4,5) test split.

Runs in the FwS eval venv (``scripts/fws/setup_eval.sh``) with their repo importable, so
answers are graded by their own ``parse_answer`` / ``grade_answer``. Prompts and sampling
settings match their ``eval_math.py`` (chat template, 1856 new tokens, EOS stop); only the
number of samples per problem and the temperature vary.

  python sample_eval.py sample --fws <repo> --model <m> --temperature 0.6 --n 64 --out a.jsonl
  python sample_eval.py score  --fws <repo> --model <m> --samples a.jsonl --out a.<m>.jsonl
  python sample_eval.py grade  --fws <repo> --samples a.jsonl --out a.graded.jsonl
"""

import argparse
import json
import os
import signal
import sys
from multiprocessing import Pool

MATH_PROMPT = "You are an expert mathematician. Can you solve the following math problem? "
MATH_COT = "\n\nExplain your reasoning, thinking step by step, and return your final answer within \\boxed{{}}."
MAX_TOKENS = 1856
GRADE_TIMEOUT_SECONDS = 10


def load_problems(fws: str) -> list[dict]:
    """Their test split with the question extraction of ``eval_math.py``."""
    import pandas as pd

    sys.path.insert(0, fws)
    from grader_utils.math_parse_utils import parse_answer_from_tag

    frame = pd.read_parquet(os.path.join(fws, "math_data/test.parquet"))
    problems = []
    for idx, (wrapped, solution) in enumerate(zip(frame["question"], frame["answer"])):
        start = wrapped.index("\nUser: ") + len("\nUser: ")
        end = wrapped.rfind("Show your work")
        problems.append(
            {
                "idx": idx,
                "question": MATH_PROMPT + wrapped[start:end] + MATH_COT,
                "gold": parse_answer_from_tag(solution),
            }
        )
    return problems


def sample(args) -> None:
    from vllm import LLM, SamplingParams

    problems = load_problems(args.fws)
    llm = LLM(model=args.model, gpu_memory_utilization=0.9, seed=args.seed)
    tokenizer = llm.get_tokenizer()
    if args.raw_prompt:
        prompts = [p["question"] for p in problems]
    else:
        prompts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": p["question"]}], tokenize=False, add_generation_prompt=True
            )
            for p in problems
        ]
    params = SamplingParams(
        n=args.n,
        temperature=args.temperature,
        max_tokens=MAX_TOKENS,
        stop_token_ids=[tokenizer.eos_token_id],
        logprobs=0,
        seed=args.seed,
    )
    outputs = llm.generate(prompts, params)
    with open(args.out, "w") as f:
        for problem, output in zip(problems, outputs, strict=True):
            samples = []
            for completion in output.outputs:
                token_ids = list(completion.token_ids)
                logprob = sum(step[token].logprob for step, token in zip(completion.logprobs, token_ids, strict=True))
                samples.append(
                    {
                        "text": completion.text,
                        "token_ids": token_ids,
                        "logprob": logprob,
                        "truncated": completion.finish_reason == "length",
                    }
                )
            f.write(
                json.dumps({**problem, "prompt_token_ids": list(output.prompt_token_ids), "samples": samples}) + "\n"
            )


def score(args) -> None:
    """Sequence log-likelihood of every sampled completion under ``--model`` (prefill)."""
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    rows = [json.loads(line) for line in open(args.samples)]
    llm = LLM(model=args.model, gpu_memory_utilization=0.9)
    sequences, owners = [], []
    for row_index, row in enumerate(rows):
        for sample_index, s in enumerate(row["samples"]):
            if args.only_correct and not s["correct"]:
                continue
            sequences.append(TokensPrompt(prompt_token_ids=row["prompt_token_ids"] + s["token_ids"]))
            owners.append((row_index, sample_index))
    outputs = llm.generate(sequences, SamplingParams(max_tokens=1, prompt_logprobs=0))
    scores: dict[tuple[int, int], float] = {}
    for (row_index, sample_index), output in zip(owners, outputs, strict=True):
        start = len(rows[row_index]["prompt_token_ids"])
        token_ids = output.prompt_token_ids
        scores[(row_index, sample_index)] = sum(
            output.prompt_logprobs[position][token_ids[position]].logprob for position in range(start, len(token_ids))
        )
    with open(args.out, "w") as f:
        for row_index, row in enumerate(rows):
            values = [scores.get((row_index, i)) for i in range(len(row["samples"]))]
            f.write(json.dumps({"idx": row["idx"], "logprob": values}) + "\n")


def _timeout(signum, frame):
    raise TimeoutError


def _grade_one(item: tuple[str, str]) -> int:
    from grader_utils.math_grader import grade_answer
    from grader_utils.math_parse_utils import parse_answer

    text, gold = item
    signal.signal(signal.SIGALRM, _timeout)
    signal.alarm(GRADE_TIMEOUT_SECONDS)
    try:
        return int(grade_answer(parse_answer(text), gold))
    except Exception:  # their safe_grade: any grader failure counts as incorrect
        return 0
    finally:
        signal.alarm(0)


def grade(args) -> None:
    sys.path.insert(0, args.fws)
    rows = [json.loads(line) for line in open(args.samples)]
    items = [(s["text"], row["gold"]) for row in rows for s in row["samples"]]
    with Pool(args.workers, initializer=sys.path.insert, initargs=(0, args.fws)) as pool:
        correct = iter(pool.map(_grade_one, items, chunksize=64))
    with open(args.out, "w") as f:
        for row in rows:
            for s in row["samples"]:
                s["correct"] = next(correct)
            f.write(json.dumps(row) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("sample")
    p.add_argument("--fws", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--temperature", type=float, required=True)
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--raw-prompt", action="store_true", help="the question as plain text, without the chat template")
    p.add_argument("--out", required=True)
    p = sub.add_parser("score")
    p.add_argument("--fws", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--samples", required=True)
    p.add_argument("--only-correct", action="store_true")
    p.add_argument("--out", required=True)
    p = sub.add_parser("grade")
    p.add_argument("--fws", required=True)
    p.add_argument("--samples", required=True)
    p.add_argument("--workers", type=int, default=os.cpu_count())
    p.add_argument("--out", required=True)
    args = parser.parse_args()
    {"sample": sample, "score": score, "grade": grade}[args.command](args)


if __name__ == "__main__":
    main()
