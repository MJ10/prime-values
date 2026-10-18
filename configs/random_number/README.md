# Random integer generation: reward maximization vs distribution matching

A replication of the motivating experiment in Hu et al., *Amortizing
intractable inference in large language models* ([arXiv:2310.04363](https://arxiv.org/abs/2310.04363),
Section 2). The model is prompted with

```text
The following is a random integer drawn uniformly between 0 and 100: 
```

(trailing space included, no chat markup). The paper reports that GPT-J 6B
emits a valid number only 50.5% of the time, with a KL of 3.37 to uniform. PPO
on a validity reward reaches 95.8% valid numbers but stays highly skewed,
because every valid number earns the same reward and so there is no gradient
toward uniformity. GFlowNet fine-tuning reaches 100% valid numbers with a KL
of about 1e-4.

## Setup

`base.toml` holds everything the arms share:

- `Qwen/Qwen2.5-1.5B-Instruct` with full fine-tuning (the paper used an
  instruction-tuned GPT-J 6B with LoRA). Qwen tokenizes digits one at a time,
  so every number has exactly one token sequence. Modern Qwen models are much
  closer to uniform on this prompt than GPT-J (KL 3.37). Over 4,000 samples,
  counting any whitespace/punctuation token as a terminator:

  | Model | Valid | KL to uniform | Most frequent |
  | --- | ---: | ---: | --- |
  | Qwen3-1.7B-Base | 96.2% | 0.055 | 75, 50, 25 |
  | Qwen3-1.7B | 96.5% | 0.259 | 42, 47, 37 |
  | Qwen2.5-1.5B | 95.4% | 0.301 | 45, 42, 50 |
  | Qwen3-4B-Instruct-2507 | 99.8% | 0.188 | 32, 34, 37 |
  | Qwen2.5-1.5B-Instruct | 96.3% | 0.472 | 45, 42, 43 |

  Instruction tuning adds skew, as the paper suggests. The study uses the most
  skewed model. Sampling noise alone gives a KL of about 0.013 at this sample
  size.
- The `random-number` env (`environments/random_number`), with one stop action
  like the paper's EOS. The terminator is `"."`, not EOS:
  Qwen2.5-1.5B-Instruct ended 1 of 10,302 base samples with EOS after a
  number, while it ends 73% of its numbers with a lone `"."` token. Over 512k
  samples the base model is 70.4% valid with a KL of 0.53, against GPT-J's
  50.5% and 3.37 in the paper.
  Generation stops at the first token containing `"."`, so a rollout can end
  right after any digit, like the paper's GFlowNet, which can terminate at
  every state. A rollout is valid when it is exactly the digit tokens of a
  canonical integer in `[0, 100]` (no leading zeros) plus one `"."` token.
  Merged tokens such as `".\n"` and `".\n\n"` also stop generation but are
  invalid, so every number has exactly one valid completion.
- The reward is `exp(log R)`, following the paper's reward:
  - `log R = 0` for a valid rollout.
  - `log R = -8 - 8 * (1 + 2 + ... + L)` for an invalid rollout of `L` tokens.
    This is the paper's per-position penalty, measured against its +8 for a
    valid number.

  Invalid rewards are at most `e^-16`, so grpo and value see what is in effect
  a 0/1 validity reward. The `valid_number` metric is the exact 0/1 validity,
  and `number` is the parsed value, or -1 when the rollout is invalid.
- Sampling at temperature 1, with at most 5 completion tokens, the paper's
  `max_len` (a valid rollout has at most 4). 256 rollouts per step for 300
  steps, with lr 2e-6.
- One GPU for inference and one for the trainer. The value arm adds a value
  trainer and a value evaluator, for four GPUs in total.

## Arms

| Arm | Credit | Group size | Paper analogue |
| --- | --- | ---: | --- |
| `base_model` | none (lr 0, 40 steps) | 16 | base model |
| `grpo` | `R - mean(R_group)` | 16 | reward maximization |
| `value` | `R - V_t` (learned critic, GAE) | 1 | PPO |
| `gfn` | `(log R - log π) - mean(group)` | 16 | GFlowNet fine-tuning |

`gfn` is on-policy trajectory balance, with the group mean of `log R - log π`
standing in for `log Z`. Its expected gradient is the reverse-KL gradient
toward `π ∝ R`, so it stops only once valid numbers are uniform.

The target needs the invalid completions to carry negligible total mass.
There are up to `V^L` of them at length `L` (vocab `V ≈ 152k`). The quadratic
penalty brings their total weight to `Σ_L V^L e^(-8 - 4L(L+1)) ≈ 0.017`,
against 101 for the valid numbers, so about 0.02% of the target is invalid,
as in the paper. A flat reward floor fails here: with `min_reward = 1e-10` and
6-token completions, invalid mass was about `1e21`. That `gfn` run held a KL
of 0.03–0.05 to uniform for about 220 steps, then drifted into gibberish and
reached 0% valid. `gfn`'s `min_reward = 1e-60` sits below the smallest env
reward (`e^-128`), so it only catches rollouts that failed with reward 0.

`grpo` and `value` start from about 70% validity. Once validity saturates,
every advantage is zero: this is the paper's zero-gradient argument. By default
the orchestrator filters zero-advantage rollouts and stops a run after 10
consecutive empty batches, which would leave it without final weights. Both
arms therefore ship zero-advantage rollouts and train to `max_steps` like
`gfn`.

## Launch

The runs are small, so Mila is the default. Use L40S nodes: on 2026-10-18,
jobs on A100L nodes `cn-g028` and `cn-g029` failed with BeeGFS I/O errors (see
the start-run skill). Steps average about 27 s, so a 300-step arm takes about
2.5 hours.

Mila's `main` QOS allows 2 GPUs per user, which is enough for the short
`base_model` arm. Run the other arms on `long`, which is preemptible. The
configs checkpoint every 50 steps and resume after a requeue.

`cluv submit` blocks until its job starts, so submit in the background. Space
the submissions apart so their remote `git pull` calls don't overlap:

```bash
uv run scripts/cluv_prime_rl.py submit mila --gpus-per-node=l40s:2 --time=1:00:00 \
  -- rl @ configs/random_number/base.toml @ configs/random_number/arms/base_model.toml \
  --wandb.name base_model > cluv-submit-base_model.log 2>&1 &
sleep 60
for arm in grpo gfn; do
  uv run scripts/cluv_prime_rl.py submit mila --partition=long --gpus-per-node=l40s:2 --time=6:00:00 \
    -- rl @ configs/random_number/base.toml @ configs/random_number/arms/$arm.toml \
    --wandb.name $arm > cluv-submit-$arm.log 2>&1 &
  sleep 60
done
uv run scripts/cluv_prime_rl.py submit mila --partition=long --gpus-per-node=l40s:4 \
  --cpus-per-task=16 --mem=128G --time=6:00:00 \
  -- rl @ configs/random_number/base.toml @ configs/random_number/arms/value.toml \
  --wandb.name value > cluv-submit-value.log 2>&1 &
wait
```

Jobs read the shared remote checkout when they start. Don't sync a new commit
to the cluster while any arm is still queued.

## Analysis

`uv run scripts/cluv_prime_rl.py sync mila` fetches each run to
`~/scratch/prime-values/mila_<job-id>/`. Then run:

```bash
uv run scripts/random_number_report.py \
  base=~/scratch/prime-values/mila_<id> grpo=... value=... gfn=... \
  --out outputs/random_number_report
```

The script pools the last 10 steps of each arm (2,560 samples) and reports:

- the valid fraction
- KL from the valid numbers' empirical distribution to uniform on `[0, 100]`
- entropy and the number of distinct values

It writes Figure 2-style log-scale histograms and per-step curves. With 2,560
samples, an exactly uniform sampler still shows an empirical KL of about
`100 / (2 × 2560) ≈ 0.02`.

### Figure 2 from large samples

The paper's Figure 2 draws 512k samples from each final model.
`scripts/random_number_sample.py` does the same with vLLM. It uses the env's
prompt, terminator and parser, with every sampling knob set explicitly. Submit
one single-GPU job per model. Pass `Qwen/Qwen2.5-1.5B-Instruct` for the base
model, or a run's final `weights/step_<n>` directory on the cluster as an
absolute path:

```bash
uv run scripts/cluv_prime_rl.py submit mila --gpus-per-node=l40s:1 --time=1:00:00 \
  -- python scripts/random_number_sample.py /network/scratch/.../mila_<id>/weights/step_<n> \
  --num-samples 512000 > cluv-submit-sample.log 2>&1 &
```

Each job writes `numbers.json` to its own run directory. After a sync, pass
those files to the report as `label=.../numbers.json`; it pools all of their
samples instead of the last 10 steps.
