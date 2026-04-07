"""
Loss functions for StratifiedDQE on AirfRANS.

Loss inventory
--------------
airfrans_loss         — surface-weighted MSE for CFD node regression
stratum_entropy_loss  — maximise per-sample stratum entropy (avoid collapse)
curvature_diversity   — push stratum curvatures apart
centripetal_loss      — pull embeddings toward stratum prototypes
"""

from __future__ import annotations
import math
import torch
import torch.nn.functional as F


# ── CFD regression ─────────────────────────────────────────────────────────

def airfrans_loss(
    pred:         torch.Tensor,   # (N, T) — predicted targets
    target:       torch.Tensor,   # (N, T) — true targets
    surf_mask:    torch.Tensor,   # (N,)   — bool, True on airfoil surface
    surf_weight:  float = 10.0,
) -> torch.Tensor:
    """Weighted MSE: surface nodes get `surf_weight`× more importance.

    AirfRANS surface nodes carry boundary-layer physics that matters most
    for aerodynamic coefficients, so they deserve a higher loss weight.
    """
    per_node = F.mse_loss(pred, target, reduction="none").mean(dim=-1)  # (N,)
    w = torch.where(surf_mask, torch.full_like(per_node, surf_weight), torch.ones_like(per_node))
    return (w * per_node).sum() / w.sum()


# ── Stratum regularisers ────────────────────────────────────────────────────

def stratum_entropy_loss(
    soft_assign: torch.Tensor,  # (N, K) — soft stratum weights (sum to 1)
    eps: float = 1e-8,
) -> torch.Tensor:
    """Maximise per-sample entropy over strata.

    High entropy means a node's assignment is spread across strata; a
    collapsed solution (all mass on one stratum) has zero entropy.  We
    *minimise* the negative mean entropy.

    Returns a scalar ≥ 0 (0 = maximum entropy, uniform assignment).
    """
    H = -(soft_assign * (soft_assign.clamp(min=eps)).log()).sum(dim=-1)   # (N,)
    H_max = torch.log(torch.tensor(soft_assign.shape[-1], dtype=H.dtype, device=H.device))
    return (H_max - H.mean()).clamp(min=0.0)  # penalise deviation from max entropy


def curvature_diversity_loss(
    kappas: torch.Tensor,   # (K,) — one κ per stratum
    margin: float = 0.1,
) -> torch.Tensor:
    """Encourage strata to occupy different curvature values.

    For each pair (i, j), penalise |κ_i − κ_j| < margin.  This is a
    soft hinge: penalty = max(0, margin − |κ_i − κ_j|).
    """
    K = kappas.shape[0]
    if K < 2:
        return kappas.new_zeros(())
    ki = kappas.unsqueeze(1).expand(K, K)   # (K, K)
    kj = kappas.unsqueeze(0).expand(K, K)   # (K, K)
    diff = (ki - kj).abs()                   # (K, K)
    # upper triangle only (pairs counted once)
    mask = torch.triu(torch.ones(K, K, device=kappas.device, dtype=torch.bool), diagonal=1)
    return F.relu(margin - diff[mask]).mean()


def centripetal_loss(
    embeds:      torch.Tensor,   # (N, d) — embeddings in tangent space (log_0)
    protos:      torch.Tensor,   # (K, d) — prototype locations in tangent space
    soft_assign: torch.Tensor,   # (N, K) — soft stratum assignments
) -> torch.Tensor:
    """Pull each embedding towards its expected prototype.

    Expected prototype for node i: p̄_i = Σ_k w_{ik} · proto_k

    Loss = mean ‖embed_i − p̄_i‖²
    """
    expected_proto = soft_assign @ protos           # (N, d)
    return F.mse_loss(embeds, expected_proto)


# ── Stratum specialization losses ──────────────────────────────────────────

def prediction_diversity_loss(
    preds_stack: torch.Tensor,  # (N, K, T) — per-stratum predictions
) -> torch.Tensor:
    """Option 1 — penalise cosine similarity between mean per-stratum predictions.

    Pushes each stratum's head to learn a distinct predictive function.
    Returns scalar ≥ 0; 0 = mean predictions are mutually orthogonal.

    Gradient flows back into the DQE heads, so they are actively trained to
    diverge.  The regression loss prevents the trivial solution of predicting
    zero (which has undefined cosine similarity and no gradient).
    """
    K = preds_stack.shape[1]
    if K < 2:
        return preds_stack.new_zeros(())
    mu      = preds_stack.mean(dim=0).float()          # (K, T)
    mu_norm = F.normalize(mu, dim=-1)                   # (K, T)
    sim     = mu_norm @ mu_norm.T                       # (K, K) cosine similarity
    mask    = torch.triu(
        torch.ones(K, K, dtype=torch.bool, device=sim.device), diagonal=1
    )
    return F.relu(sim[mask]).mean()                     # hinge at 0, upper-tri pairs


def conditional_diversity_loss(
    preds_stack: torch.Tensor,  # (N, K, T) — per-stratum predictions
    soft_assign: torch.Tensor,  # (N, K)    — soft stratum weights
) -> torch.Tensor:
    """Option 3 — jointly reward concentrated assignments AND diverse predictions.

    Two terms summed with equal weight:
      • Assignment concentration: minimise per-node assignment entropy so each
        node is "owned" by one stratum.  Normalised to [0, 1] by log(K).
      • Prediction diversity: same pairwise cosine hinge as option 1.

    ⚠ Conflicts with stratum_entropy_loss, which maximises entropy.  Set
    entropy_weight=0 in total_loss when using this mode to avoid opposing
    gradients.
    """
    K   = preds_stack.shape[1]
    eps = 1e-8

    # Concentration term — penalise high per-node assignment entropy
    H        = -(soft_assign * soft_assign.clamp(eps).log()).sum(-1).mean()
    H_norm   = H / max(math.log(K), eps)   # in [0, 1]

    # Diversity term — penalise inter-stratum prediction similarity
    if K >= 2:
        mu      = preds_stack.mean(dim=0).float()      # (K, T)
        mu_norm = F.normalize(mu, dim=-1)
        sim     = mu_norm @ mu_norm.T                   # (K, K)
        mask    = torch.triu(
            torch.ones(K, K, dtype=torch.bool, device=sim.device), diagonal=1
        )
        sim_term = F.relu(sim[mask]).mean()
    else:
        sim_term = preds_stack.new_zeros(())

    return H_norm + sim_term


# ── Combined training objective ─────────────────────────────────────────────

def total_loss(
    pred:           torch.Tensor,
    target:         torch.Tensor,
    surf_mask:      torch.Tensor,
    soft_assign:    torch.Tensor,
    embeds_tan:     torch.Tensor,
    protos_tan:     torch.Tensor,
    kappas:         torch.Tensor,
    *,
    surf_weight:    float = 10.0,
    entropy_weight: float = 0.1,
    diversity_weight: float = 0.05,
    centripetal_weight: float = 0.01,
    curvature_margin: float = 0.1,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Weighted sum of all losses.

    Returns
    -------
    loss   : scalar tensor (backward-able)
    breakdown : {name: float} for logging
    """
    l_reg  = airfrans_loss(pred, target, surf_mask, surf_weight)
    l_ent  = stratum_entropy_loss(soft_assign)
    l_div  = curvature_diversity_loss(kappas, curvature_margin)
    l_cent = centripetal_loss(embeds_tan, protos_tan, soft_assign)

    loss = (l_reg
            + entropy_weight     * l_ent
            + diversity_weight   * l_div
            + centripetal_weight * l_cent)

    breakdown = {
        "regression":  l_reg.item(),
        "entropy":     l_ent.item(),
        "diversity":   l_div.item(),
        "centripetal": l_cent.item(),
        "total":       loss.item(),
    }
    return loss, breakdown
