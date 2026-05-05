"""Plackett–Luce and robust Plackett–Luce losses for listwise DPO.

Math is the source of truth in
``vibecoding/ROBUST_LISTWISE_DPO_MATH.md`` and
``vibecoding/online_robust_listwise_dpo_implementation_guide.md``.
"""

from __future__ import annotations

import torch


def worst_case_ranking(g: torch.Tensor) -> torch.Tensor:
    """``sigma_wc = argsort(g, ascending)`` (highest score placed last)."""
    return torch.argsort(g, dim=-1, descending=False)


def plackett_luce_loss(
    g_in_rank_order: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """Plackett–Luce NLL of a ranking.

    Args:
        g_in_rank_order: ``[B, K]`` scores already reordered so that column
            ``i`` holds the score of the candidate ranked at position ``i``.
        reduction: ``"mean"`` (default, scalar) or ``"none"`` (``[B]``).
    """
    flipped = torch.flip(g_in_rank_order, dims=[-1])
    cumlse_flipped = torch.logcumsumexp(flipped, dim=-1)
    suffix_lse = torch.flip(cumlse_flipped, dims=[-1])  # [B, K]
    nll = (suffix_lse - g_in_rank_order).sum(dim=-1)  # [B]
    if reduction == "none":
        return nll
    if reduction == "mean":
        return nll.mean()
    raise ValueError(f"unknown reduction {reduction!r}")


def robust_pl_loss(
    g: torch.Tensor,
    ranking_obs: torch.Tensor,
    rho: float,
    reduction: str = "mean",
) -> torch.Tensor:
    """``(1 - rho) * PL(g, sigma_obs) + rho * PL(g, sigma_wc)``.

    Args:
        g: ``[B, K]`` raw scores (e.g. ``g_theta``).
        ranking_obs: ``[B, K]`` long tensor; column ``i`` holds the index of
            the candidate placed at rank ``i`` under the observed ordering.
        rho: scalar in [0, 1].
        reduction: ``"mean"`` (default, scalar) or ``"none"`` (``[B]``).
    """
    g_obs = g.gather(1, ranking_obs)
    nominal = plackett_luce_loss(g_obs, reduction=reduction)
    if rho == 0.0:
        return nominal
    sigma_wc = worst_case_ranking(g)
    g_wc = g.gather(1, sigma_wc)
    worst = plackett_luce_loss(g_wc, reduction=reduction)
    return (1.0 - rho) * nominal + rho * worst
