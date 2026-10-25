# GFlowNet fine-tuning vs Finetuning with Sampling (math, Qwen2.5-3B)

A GFlowNet baseline in the math setup of Karan, Chen and Du, *Finetuning with
Sampling: SFT Learns Better Than You Think* ([arXiv:2610.02140](https://arxiv.org/abs/2610.02140),
code at [aakaran/finetuning-with-sampling](https://github.com/aakaran/finetuning-with-sampling),
pinned here at `e74ab5e`). FwS turns expert solutions into samples of the
information projection

```text
p_C(x) ∝ p_base(x | q) · 1[x ∈ C]      (C = correct traces)
```

with a block-wise MCMC sampler, then runs plain SFT on those samples. That is a
GFlowNet target: `log R(x) = log p_base(x | q) + log 1[x correct]`. This study
trains toward it directly with `gfn` (on-policy trajectory balance, Hu et al.,
[arXiv:2310.04363](https://arxiv.org/abs/2310.04363)) and evaluates every model
with FwS's own scripts.

## What FwS's release does

Read from their repository (treated as untrusted code: only their eval scripts
run, in a separate venv):

- **Prompt.** One user turn, `MATH_PROMPT + problem + MATH_COT`, through the
  tokenizer's chat template with `add_generation_prompt=True`. Qwen2.5-3B's
  template adds the system prompt "You are a helpful assistant.". `MATH_COT` is
  never `.format`-ed, so the model literally sees `\boxed{{}}`.
- **Eval sampling.** vLLM, temperature 0.6, top-p 1, 1856 new tokens, stop on
  EOS (`<|endoftext|>`) only, one sample per problem. `eval_math.py` sets no
  seed; `eval_ood_math.py` uses `seed=0`.
- **Grading.** The last `\boxed{}` (`parse_answer`) checked by
  `grade_answer`, which is OpenAI's prm800k grader verbatim. The gold answer
  is the `<answer>` tag of the test split, and for the OOD sets the dataset's
  answer.
- **Data.** `math_data/train.parquet` has 7962 MATH L3–5 problems with expert
  solutions; `test.parquet` has the 1024 test problems. `ood_data/` holds
  AMC (83), MATH-TTT (the 500 MATH500 problems) and GSM8K (1319).
- **Released boosted traces.** `sampling_traces_math_qwen.jsonl` has 8229
  traces. 7961 are for `train.parquet` problems; the other 268 are MATH500
  problems (exact problem-text matches), none in their test split. The
  paper's "8230 training problems" is `train.parquet` plus these. So Sampling
  SFT, as released, trained on 268 of the 500 MATH500 test problems. A
  further 99 MATH500 problems are in their MATH(3,4,5) test split, and none
  in `train.parquet`.
- **Sampler.** `boost_math.py` does not implement the Metropolis–Hastings
  rule in the paper (Algorithm 1). It accepts a proposal exactly when its
  *length-normalized* mean token log-likelihood under the base model is
  higher (Appendix C.2 calls this an approximation). There is no
  proposal-ratio term and no randomness in acceptance, so it is a stochastic
  hill-climb toward high per-token likelihood rather than a sampler of `p_C`.
  The proposal is the base model prompted with the problem, expert solution
  and partial response, at temperature 0.6 with repetition penalty 1.1. The
  traces are not filtered by correctness (95.3% are correct).
- **Missing pieces.** The release omits `grader_utils/math_normalize.py`
  (imported by the grader) and the `format_prompt` helper that `eval_math.py`
  and `eval_ood_math.py` call. `scripts/fws/setup_eval.sh` restores the first
  from prm800k and the second verbatim from their `boost_math.py`;
  `format_prompt` only feeds a logged likelihood, not grading.

## Setup here

- **Env** `fws-math` (`environments/fws_math`): `train.parquet`, their prompt,
  gold = last boxed answer of the expert solution (as `boost_math.py` does).
  Two solutions write `\boxed 2` without braces; FwS's parser cannot read
  them either, so they are dropped (7960 problems).
  The prompt tokens are identical to FwS's `apply_chat_template` call. The
  reward is math-verify on the last `\boxed{}`. On the 7961 released traces
  math-verify marks 95.28% correct against FwS's label of 95.35%.
- **Sample space.** Temperature 1, 1856 new tokens, EOS-only stop: the same
  space as FwS's eval generations, so the target is defined over what is
  evaluated.
- **Budget.** 250 steps × 64 prompts × 8 rollouts. That is 16,000 prompt
  visits (~2 epochs of 7960, matching FwS's 2-epoch SFT) and 128,000
  sampled traces. Full fine-tuning, 1 inference + 1 trainer H100.

## Arms

| Arm | Signal | Target |
| --- | --- | --- |
| `grpo` | `R − mean_group(R)` | max reward |
| `gfn` | `log 1_ε[correct] + log p_base(x) − log π(x)`, centered | `p_C` |
| `gfn_offpolicy` | as `gfn`, plus FwS's boosted trace as a 9th group member | `p_C` |
| `sft/expert` | CE on expert solutions | — (FwS's "SFT") |
| `sft/boosted_released` | CE on all released boosted traces | FwS's Sampling SFT |
| `sft/boosted_clean` | CE on the 7961 non-MATH500 traces | Sampling SFT, decontaminated |

`gfn` prefill-scores every sample under a frozen Qwen2.5-3B
(`[orchestrator.algo.prior]`), served on the inference GPU by
`scripts/fws/rl_with_prior.sh`. With the prior, the ε floor on the indicator
is safe. `p_base` normalizes the incorrect space, so incorrect traces hold
`ε(1 − acc)/acc` of the target mass, unlike the flat floor over `V^L`
sequences that broke the random-number run.

The off-policy arm uses the group mean over on- and off-policy members. This
is the gradient of the log-variance (VarGrad) form of trajectory balance,
which is valid for any behavior distribution with the same minimizer.

## Launch

The runs used Rorqual. Its compute nodes cannot reach Hugging Face, so
download the model on the login node first. Prepare the data there too:

```bash
bash scripts/fws/setup_eval.sh "$SCRATCH/fws"
uv run python scripts/fws/offpolicy_traces.py "$SCRATCH/fws/finetuning-with-sampling" "$SCRATCH/fws/offpolicy_boosted.jsonl"
uv run python scripts/fws/sft_data.py "$SCRATCH/fws/finetuning-with-sampling" "$SCRATCH/fws/sft"
```

```bash
uv run scripts/cluv_prime_rl.py submit rorqual --time=12:00:00 \
  -- bash scripts/fws/rl_with_prior.sh @ configs/fws_gfn/base.toml @ configs/fws_gfn/arms/gfn.toml --wandb.name gfn
uv run scripts/cluv_prime_rl.py submit rorqual --time=12:00:00 \
  -- rl @ configs/fws_gfn/base.toml @ configs/fws_gfn/arms/grpo.toml --wandb.name grpo
uv run scripts/cluv_prime_rl.py submit rorqual --time=3:00:00 \
  -- sft @ configs/fws_gfn/sft/base.toml @ configs/fws_gfn/sft/boosted_clean.toml @ configs/fws_gfn/sft/bf16.toml
uv run scripts/cluv_prime_rl.py submit rorqual --time=4:00:00 \
  -- bash scripts/fws/eval_model.sh <scratch>/fws <run>/weights/step_<N> <out> <scratch>/fws/results/base/suite
```

Pass cluster paths literally: cluv does not expand `$SCRATCH` in program
arguments. Pass scripts, never inline `bash -c`, because cluv's remote
quoting mangles inline shell. Jobs hang in W&B's final upload on Rorqual;
cancel them once `weights/step_<N>/STABLE` exists.

## Evaluation

- `scripts/fws/eval_fws.sh <fws_dir> <model> <out>`: FwS's `eval_math.py` and
  `eval_ood_math.py` unchanged, one shard per GPU. This gives the
  single-shot MATH(3,4,5), AMC, MATH500 and GSM8K numbers.
- `scripts/fws/sample_suite.sh <fws_dir> <model> <out> [<base_out>]` and
  `scripts/fws/report.py` produce:
  - pass@k (k ≤ 64) and mean accuracy at temperature 0.6, graded by FwS's
    grader;
  - at temperature 1, `KL(π ‖ p_C)` (from `log Z_q` estimated with the base
    model's own samples);
  - `KL(p_C ‖ π)` (the base model's correct samples are exact samples of
    `p_C`).

## Results

All numbers are single-shot accuracy from FwS's own `eval_math.py` and
`eval_ood_math.py` (temperature 0.6, 1856 tokens), unless a column says
otherwise. FwS's numbers are their Table 1. Sampler columns are on
MATH(3,4,5) with FwS's grader: mean accuracy over 64 samples at 0.6, pass@64,
and the KLs at temperature 1, in nats over problems the base model solves.

| Model | MATH(3,4,5) | AMC | MATH500 | GSM8K | mean@64 | pass@64 | KL(π‖p_C) | KL(p_C‖π) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Base (FwS Table 1) | 0.315 | 0.133 | 0.245 | 0.579 | | | | |
| SFT (FwS Table 1) | 0.243 | 0.100 | 0.168 | 0.455 | | | | |
| GRPO (FwS Table 1) | 0.457 | 0.249 | 0.313 | 0.808 | | | | |
| Sampling SFT (FwS Table 1) | 0.495 | 0.277 | 0.582 | 0.782 | | | | |
| Sampling SFT + RL (FwS Table 1) | 0.545 | 0.241 | 0.652 | 0.830 | | | | |
| **Base, reproduced** | **0.462** | **0.193** | **0.522** | **0.652** | 0.439 | 0.888 | 4.6 | 1.4 |
| GRPO (ours) | 0.634 | 0.373 | 0.648 | 0.842 | 0.618 | 0.887 | 37.0 | 119.1 |
| **GFN, base prior, on-policy (ours)** | **0.545** | **0.265** | **0.600** | **0.815** | 0.539 | 0.896 | 3.8 | 1.8 |
| GFN, base prior, + FwS boosted traces off-policy (ours) | 0.541 | 0.289 | 0.568 | 0.815 | | | | |
| SFT on expert solutions (ours, fp32, lr 5e-5) | 0.218 | 0.072 | 0.250 | 0.458 | 0.210 | 0.711 | 76.8 | 325.0 |
| Sampling SFT, released traces, bf16 (ours, FwS recipe) | 0.510 | 0.205 | 0.604 | 0.808 | 0.497 | 0.882 | 34.1 | 73.5 |
| Sampling SFT, clean traces, bf16 (ours, FwS recipe) | 0.503 | 0.217 | 0.570 | 0.777 | 0.495 | 0.891 | 34.0 | 73.2 |
| Sampling SFT, released traces, fp32, lr 5e-5 | 0.341 | 0.145 | 0.482 | 0.670 | 0.357 | 0.817 | 62.7 | 150.5 |
| Sampling SFT, clean traces, fp32, lr 5e-5 | 0.363 | 0.157 | 0.408 | 0.673 | 0.360 | 0.824 | 63.3 | 150.4 |
| Sampling SFT, clean traces, fp32, lr 1e-5 | 0.501 | 0.229 | 0.548 | 0.783 | 0.509 | 0.885 | 36.7 | 88.5 |
| Sampling SFT, clean traces, fp32, lr 5e-6 | 0.542 | 0.229 | 0.582 | 0.792 | 0.524 | 0.896 | 30.6 | 61.5 |

Pending: gfn_offpolicy sampler columns.

### Caveats

- **Base model.** With their eval script, environment pins and grader, base
  Qwen2.5-3B scores 0.462 on MATH(3,4,5), not 0.315, and is higher on every
  column. math-verify agrees with their grader on 99.3% of the 1024
  problems. Without the chat template it still scores 0.423 (4 samples at
  0.6). The gap is unexplained, so gains over their base row are overstated
  by the gap.
- **MATH500 contamination.** Their released Sampling-SFT data includes 268
  of the 500 MATH500 problems.
- **Checkpoint configs.** Checkpoints are written by transformers 5, which
  moved `rope_theta` into `rope_parameters`. FwS's eval venv (transformers
  4.57) silently reads such a config with RoPE base 1e4 instead of Qwen's
  1e6, so `scripts/fws/eval_model.sh` evaluates every checkpoint with the base
  model's config and tokenizer files. Training is unaffected.
- **SFT precision.** FwS's verl SFT (`model_dtype=bf16`) updates bf16
  weights with no fp32 master copy, so at lr 5e-5 many updates round away.
  prime-rl keeps fp32 master weights. The `sft/bf16.toml` overlay
  matches their precision for the Sampling-SFT reproduction only; every
  other arm uses the default fp32 optimization dtype. The fp32 lr sweep is
  kept as a cross-check.
- **SFT batching.** prime-rl packs SFT samples into 2048-token rows, so a step
  holds about 16 samples' worth of tokens rather than exactly 16 samples.
- **Learning rate.** `gfn` uses grpo's lr (1e-6). Adam normalizes gradient
  scale, so the larger advantage scale does not change the step size.
- **Empty completions.** Rollouts that end immediately with EOS are dropped
  as errors by verifiers, so they are excluded from gfn's sample space. This
  is rare.
