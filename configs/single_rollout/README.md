# Single-rollout vs group-rollout RL on MATH

How close can one rollout per prompt with a learned critic baseline get to
GRPO's group-mean baseline at the same rollout budget?

## Setup

`base.toml` holds everything shared by the arms:

- `Qwen/Qwen3-0.6B` in non-thinking mode, at most 1536 completion tokens
- training on Hendrycks MATH (v0 `math-env`), binary math-verify reward
- MATH-500 avg@4 at startup and every 25 steps, sampled like training
  (temperature 1.0, thinking disabled)
- 256 rollouts per step for 200 steps
- one GPU each for policy inference, the policy trainer, the value trainer,
  and the value evaluator

Group arms also train the critic, which serves as a diagnostic: its
`value/explained_variance` and `value/rollout_*` metrics show how well a learned
baseline would track the group mean on the same data.

## Arms

| Arm | Prompts/step | Rollouts/prompt | Credit |
| --- | ---: | ---: | --- |
| `g8_mean` | 32 | 8 | `R - mean(R_group)` |
| `g4_mean` | 64 | 4 | `R - mean(R_group)` |
| `g2_mean` | 128 | 2 | `R - mean(R_group)` |
| `s1_value` | 256 | 1 | `R - V_t` (GAE, lambda 1) |
| `s1_value_replay4` | 256 | 1 | `s1_value`, up to 4 critic updates per rollout |
| `s1_value_lam95` | 256 | 1 | GAE with policy lambda 0.95 |

Group arms drop prompts whose rollouts all receive the same reward (the
zero-advantage filter), so their effective batch shrinks as accuracy
saturates. Single-rollout arms always train on all 256 samples.

## Launch

Compose the base with one arm. Run every arm on the same cluster and GPU type
so step times are comparable. A 10-step smoke run on 4x H100 took about 20 s
per step plus about 2.5 min per eval, so a full arm takes roughly 1.5 hours.
On Tamia, the three-hour `gpubase_bynode_b1` partition fits one arm:

`cluv submit` waits until its job starts, so submit the arms in the
background. The stagger lets each submit finish its remote sync before the next
one runs `git pull` in the same checkout:

```bash
for arm in g8_mean g4_mean g2_mean s1_value s1_value_replay4 s1_value_lam95; do
  uv run scripts/cluv_prime_rl.py submit tamia \
    --partition=gpubase_bynode_b1 --time=3:00:00 \
    -- rl @ configs/single_rollout/base.toml @ configs/single_rollout/arms/$arm.toml \
    --wandb.name $arm > cluv-submit-$arm.log 2>&1 &
  sleep 60
done
wait
```

Jobs read the remote checkout when they start, so do not sync a new commit to
the cluster while arms are still queued.

The base config checkpoints the policy and critic every 25 steps and resumes
from the latest checkpoint when Slurm requeues the job. The critic replay
buffer refills from fresh rollouts after a resume.

## What to compare

- `eval/math500/*` against step: all arms consume the same rollouts per step.
- Training reward and the trainable-sample fraction.
- `value/explained_variance` and `value/rollout_advantage_std`: critic quality
  and advantage scale in each arm.
- Trainer and orchestrator step time.
