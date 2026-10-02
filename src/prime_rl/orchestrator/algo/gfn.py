from __future__ import annotations

import math
from typing import TYPE_CHECKING

from prime_rl.configs.algorithm import GFNAlgoConfig
from prime_rl.orchestrator.algo.advantage import group_advantages
from prime_rl.orchestrator.algo.base import Algorithm

if TYPE_CHECKING:
    from prime_rl.orchestrator.types import Rollout


def sequence_logprob(rollout: Rollout) -> float:
    """Sampling log-probability of the rollout's trainable tokens (temperature 1 is
    enforced by the orchestrator config)."""
    return math.fsum(
        token_logprob
        for sample in rollout.samples
        for trainable, token_logprob in zip(sample.mask, sample.logprobs, strict=True)
        if trainable
    )


class GFNAlgorithm(Algorithm):
    """GFlowNet fine-tuning (arXiv:2310.04363): distribution matching toward
    ``π(x) ∝ R(x)``. Each rollout's scalar advantage is ``log R(x) − log π(x)``
    minus its group mean — on-policy trajectory balance with the group mean as
    the ``log Z`` estimate. Its expected gradient is the reverse-KL gradient
    ``KL(π || R/Z)``, which vanishes only when the policy matches the target."""

    config: GFNAlgoConfig

    @property
    def minimum_group_size(self) -> int:
        return 2

    async def score_group(self, group: list[Rollout]) -> None:
        log_ratios = []
        for rollout in group:
            if rollout.reward < 0:
                raise ValueError(f"gfn needs non-negative rewards, got {rollout.reward} (env '{rollout.env_name}')")
            log_reward = math.log(max(rollout.reward, self.config.min_reward))
            log_ratios.append(log_reward - sequence_logprob(rollout))
        for rollout, advantage in zip(group, group_advantages(log_ratios, "mean"), strict=True):
            rollout.assign_advantages(advantage)
