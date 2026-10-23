from __future__ import annotations

import asyncio
import json
import math
import os
from typing import TYPE_CHECKING

from prime_rl.configs.algorithm import GFNAlgoConfig
from prime_rl.orchestrator.algo.advantage import group_advantages
from prime_rl.orchestrator.algo.base import Algorithm
from prime_rl.transport import TrainingSample

if TYPE_CHECKING:
    from prime_rl.orchestrator.types import Rollout
    from prime_rl.utils.client import InferencePool


def sequence_logprob(rollout: Rollout) -> float:
    """Sampling log-probability of the rollout's trainable tokens (temperature 1 is
    enforced by the orchestrator config)."""
    return math.fsum(
        token_logprob
        for sample in rollout.samples
        for trainable, token_logprob in zip(sample.mask, sample.logprobs, strict=True)
        if trainable
    )


def prior_logprob(rollout: Rollout) -> float:
    """Prior log-probability of the rollout's trainable tokens, from the
    prefill-scored ``ref_logprobs``."""
    return math.fsum(
        token_logprob
        for sample in rollout.samples
        for trainable, token_logprob in zip(sample.mask, sample.ref_logprobs, strict=True)
        if trainable
    )


def log_ratios(
    rewards: list[float], logprobs: list[float], prior_logprobs: list[float] | None, min_reward: float
) -> list[float]:
    """Per-trajectory ``log R(x) − log π(x)`` with
    ``log R(x) = log p_prior(x) + log max(reward, min_reward)`` (no prior term
    when ``prior_logprobs`` is None)."""
    priors = prior_logprobs if prior_logprobs is not None else [0.0] * len(rewards)
    return [
        prior + math.log(max(reward, min_reward)) - logprob
        for reward, logprob, prior in zip(rewards, logprobs, priors, strict=True)
    ]


class GFNAlgorithm(Algorithm):
    """GFlowNet fine-tuning (arXiv:2310.04363): distribution matching toward
    ``π(x) ∝ R(x)``. Each rollout's scalar advantage is ``log R(x) − log π(x)``
    minus its group mean — on-policy trajectory balance with the group mean as
    the ``log Z`` estimate. Its expected gradient is the reverse-KL gradient
    ``KL(π || R/Z)``, which vanishes only when the policy matches the target.

    With a ``prior``, ``log R`` adds the frozen prior's sequence log-likelihood,
    prefill-scored per sample like opd's teacher. With ``off_policy_path``, each
    matching group gains one off-policy trajectory before centering."""

    config: GFNAlgoConfig

    def __init__(self, config: GFNAlgoConfig, policy_pool: InferencePool):
        super().__init__(config, policy_pool)
        self.prior_pool: InferencePool | None = None  # connected in setup()
        self.off_policy: dict[str, dict] = {}
        self.eos_token_id: int | None = None
        self.tokenizer = None
        # Per-group statistics since the last ``metrics()`` call (one logging step).
        self.window: dict[str, list[float]] = {key: [] for key in self.metric_keys()}

    @property
    def minimum_group_size(self) -> int:
        return 2

    def metric_keys(self) -> list[str]:
        return ["gfn/log_z_elbo", "gfn/log_prior_ratio", "gfn/off_policy_advantage"]

    def metrics(self) -> dict[str, float]:
        """Means since the last call: ``log_z_elbo`` is the on-policy group
        mean of ``log R − log π`` (a lower bound on ``log Z``);
        ``log_prior_ratio`` is ``log p_prior − log π`` on on-policy samples
        (minus a KL(π || prior) estimate); ``off_policy_advantage`` is the
        centered log ratio of the off-policy trajectories."""
        means = {key: math.fsum(values) / len(values) for key, values in self.window.items() if values}
        self.window = {key: [] for key in self.metric_keys()}
        return means

    async def setup(self) -> None:
        if self.config.prior is not None:
            self.prior_pool = await self.connect(self.config.prior)
        if self.config.off_policy_path is not None:
            from renderers.base import load_tokenizer

            with open(os.path.expandvars(self.config.off_policy_path)) as f:
                rows = [json.loads(line) for line in f]
            self.off_policy = {row["prompt"]: row for row in rows}
            self.tokenizer = load_tokenizer(self.policy_pool.model_name)
            self.eos_token_id = self.tokenizer.eos_token_id

    async def score_rollout(self, rollout: Rollout) -> None:
        if self.prior_pool is not None:
            await self._score_prior(rollout)

    async def _score_prior(self, rollout: Rollout) -> None:
        pool = self.prior_pool
        assert pool is not None

        async def score_sample(sample: TrainingSample) -> None:
            sample.ref_logprobs = await pool.score(list(sample.token_ids))

        await asyncio.gather(*(score_sample(sample) for sample in rollout.samples))

    async def _off_policy_rollout(self, anchor: Rollout, row: dict) -> Rollout:
        """A copy of ``anchor`` whose one sample is ``anchor``'s prompt tokens
        followed by ``row``'s completion and EOS, with the live policy's prefill
        logprobs standing in for sampling logprobs."""
        assert self.tokenizer is not None and self.eos_token_id is not None
        if len(anchor.samples) != 1:
            raise ValueError(f"off-policy trajectories need single-sample rollouts, got {len(anchor.samples)}")
        anchor_sample = anchor.samples[0]
        prompt_ids = list(anchor_sample.token_ids[: anchor_sample.mask.index(True)])
        completion_ids = self.tokenizer.encode(row["completion"], add_special_tokens=False) + [self.eos_token_id]
        token_ids = prompt_ids + completion_ids
        mask = [False] * len(prompt_ids) + [True] * len(completion_ids)
        policy_logprobs = await self.policy_pool.score(token_ids)
        sample = TrainingSample(
            token_ids=token_ids,
            mask=mask,
            logprobs=[logprob if trainable else 0.0 for logprob, trainable in zip(policy_logprobs, mask, strict=True)],
            temperatures=[],
            env_name=anchor_sample.env_name,
        )
        rollout = anchor.model_copy(
            update={"samples": [sample], "rewards": {"reward": float(row["reward"])}, "advantages": None}
        )
        if self.prior_pool is not None:
            await self._score_prior(rollout)
        return rollout

    async def score_group(self, group: list[Rollout]) -> None:
        num_on_policy = len(group)
        row = self.off_policy.get(group[0].task.prompt)
        if row is not None:
            group.append(await self._off_policy_rollout(group[0], row))
        for rollout in group:
            if rollout.reward < 0:
                raise ValueError(f"gfn needs non-negative rewards, got {rollout.reward} (env '{rollout.env_name}')")
        logprobs = [sequence_logprob(rollout) for rollout in group]
        priors = None
        if self.prior_pool is not None:
            priors = [prior_logprob(rollout) for rollout in group]
            # The prior only enters the advantage; keep the wire free of a ref_kl stream.
            for rollout in group:
                for sample in rollout.samples:
                    sample.ref_logprobs = None
            self.window["gfn/log_prior_ratio"].extend(
                prior - logprob for prior, logprob in zip(priors[:num_on_policy], logprobs[:num_on_policy])
            )
        ratios = log_ratios([rollout.reward for rollout in group], logprobs, priors, self.config.min_reward)
        advantages = group_advantages(ratios, "mean")
        self.window["gfn/log_z_elbo"].append(math.fsum(ratios[:num_on_policy]) / num_on_policy)
        self.window["gfn/off_policy_advantage"].extend(advantages[num_on_policy:])
        for rollout, advantage in zip(group, advantages, strict=True):
            rollout.assign_advantages(advantage)
