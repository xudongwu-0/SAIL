"""Sanity checks for the listwise PL / robust-PL losses.

Mirrors the checks in
``vibecoding/online_robust_listwise_dpo_implementation_guide.md`` §6.1–§6.4.

Run with::

    python tests/test_listwise_losses.py
"""

from __future__ import annotations

import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cdpo.listwise_losses import (  # noqa: E402
    plackett_luce_loss,
    robust_pl_loss,
    worst_case_ranking,
)


def test_k2_reduces_to_dpo() -> None:
    torch.manual_seed(0)
    g = torch.randn(64, 2)
    ranking = torch.tensor([[0, 1]] * 64)
    pl = plackett_luce_loss(g.gather(1, ranking))
    dpo = F.softplus(g[:, 1] - g[:, 0]).mean()
    assert torch.allclose(pl, dpo, atol=1e-6), (pl.item(), dpo.item())


def test_rho0_equals_nominal() -> None:
    torch.manual_seed(1)
    g = torch.randn(32, 4)
    ranking = torch.stack([torch.randperm(4) for _ in range(32)])
    nominal = plackett_luce_loss(g.gather(1, ranking))
    rob = robust_pl_loss(g, ranking, rho=0.0)
    assert torch.allclose(nominal, rob, atol=1e-6)


def test_rho1_equals_worst_case_only() -> None:
    torch.manual_seed(2)
    g = torch.randn(32, 5)
    ranking = torch.stack([torch.randperm(5) for _ in range(32)])
    sigma_wc = worst_case_ranking(g)
    worst = plackett_luce_loss(g.gather(1, sigma_wc))
    rob = robust_pl_loss(g, ranking, rho=1.0)
    assert torch.allclose(worst, rob, atol=1e-6)


def test_worst_case_direction() -> None:
    torch.manual_seed(3)
    g = torch.randn(64, 4)
    sigma_wc = worst_case_ranking(g)
    assert (sigma_wc[:, -1] == g.argmax(dim=1)).all()
    assert (sigma_wc[:, 0] == g.argmin(dim=1)).all()


def test_pl_invariant_under_score_shift() -> None:
    torch.manual_seed(4)
    g = torch.randn(16, 4)
    ranking = torch.stack([torch.randperm(4) for _ in range(16)])
    base = plackett_luce_loss(g.gather(1, ranking))
    shifted = plackett_luce_loss((g + 7.5).gather(1, ranking))
    assert torch.allclose(base, shifted, atol=1e-5)


def test_reduction_none_matches_mean() -> None:
    torch.manual_seed(5)
    g = torch.randn(8, 4)
    ranking = torch.stack([torch.randperm(4) for _ in range(8)])
    pl_mean = plackett_luce_loss(g.gather(1, ranking))
    pl_none = plackett_luce_loss(g.gather(1, ranking), reduction="none")
    assert pl_none.shape == (8,)
    assert torch.allclose(pl_mean, pl_none.mean(), atol=1e-6)
    rob_mean = robust_pl_loss(g, ranking, rho=0.2)
    rob_none = robust_pl_loss(g, ranking, rho=0.2, reduction="none")
    assert rob_none.shape == (8,)
    assert torch.allclose(rob_mean, rob_none.mean(), atol=1e-6)


def test_sf_surrogate_gradient_matches_reinforce() -> None:
    """With lambda_sf=1 the surrogate ``stopgrad(L_b) * sum_k log p(y_{b,k})``
    has gradient = E[L_rob · Σ ∇log π(y_i)] (math doc §9).

    We mock logπ as a leaf tensor with grad and check ``sf.backward()``
    propagates ``L_b`` (per-example, mean-baseline-removed) onto it.
    """
    torch.manual_seed(6)
    B, K = 4, 3
    g = torch.randn(B, K)
    ranking = torch.stack([torch.randperm(K) for _ in range(B)])
    loss_per_ex = robust_pl_loss(g, ranking, rho=0.1, reduction="none")
    # per-token logprobs over generated y; we sum across K.
    logp_flat = torch.randn(B * K, requires_grad=True)
    sum_logp = logp_flat.view(B, K).sum(dim=1)
    adv = loss_per_ex.detach() - loss_per_ex.detach().mean()
    sf = (adv * sum_logp).mean()
    sf.backward()
    # ∂sf/∂logp_flat[b*K+k] = adv[b]/B for every k. Each row of logp_flat
    # should therefore have grad equal across its K entries.
    grads = logp_flat.grad.view(B, K)
    expected = (adv / B).unsqueeze(1).expand(B, K)
    assert torch.allclose(grads, expected, atol=1e-6)


if __name__ == "__main__":
    test_k2_reduces_to_dpo()
    test_rho0_equals_nominal()
    test_rho1_equals_worst_case_only()
    test_worst_case_direction()
    test_pl_invariant_under_score_shift()
    test_reduction_none_matches_mean()
    test_sf_surrogate_gradient_matches_reinforce()
    print("All listwise loss sanity checks passed.")
