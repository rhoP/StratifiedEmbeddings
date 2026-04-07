"""
Riemannian geometry operations for constant-curvature model spaces.

Sign convention for a stratum curvature κ
------------------------------------------
  κ < −KAPPA_EPS  →  Hyperbolic space  (Poincaré ball, radius 1/√|κ|)
  |κ| ≤ KAPPA_EPS  →  Euclidean / flat space  (degenerate limit, identity maps)
  κ > +KAPPA_EPS  →  Spherical space   (hypersphere, radius 1/√κ)

Learnable curvature
-------------------
Each stratum stores an unconstrained parameter κ_raw ∈ ℝ and maps it to a
bounded curvature via

    κ = K_max · tanh(κ_raw)   ∈ (−K_max, +K_max)

Gradients flow through tanh; the geometry *type* (sign of κ) can therefore
be learned.  The unified dispatch functions below use smooth blending rather
than hard Python branches, so gradients propagate continuously through κ even
across the flat↔hyperbolic and flat↔spherical boundaries:

    w(κ) = tanh(|κ| / GATE_SOFTNESS)   ∈ [0, 1)

    output = (1 − w) · flat_result + w · curved_result

where curved_result is selected between hyperbolic/spherical by sign(κ) via
torch.where (which is differentiable everywhere except the exact κ = 0 point,
a measure-zero event in practice).

Notation
--------
K = |κ| — positive absolute curvature.  The manifold has radius r = 1/√K.
All point tensors have shape (..., d) and operations broadcast over leading
dimensions.
"""

from __future__ import annotations
import torch
import torch.nn.functional as F

EPS          = 1e-6
KAPPA_EPS    = 5e-3   # |κ| below this → nearly flat (used only for logging)
GATE_SOFTNESS = 0.1   # temperature for flat↔curved blend; at |κ|=0.3 ≈ 95% curved


# ── helpers ────────────────────────────────────────────────────────────────

def _abs_K(kappa: torch.Tensor) -> torch.Tensor:
    """Absolute curvature K = |κ|, clamped away from zero for stability."""
    return kappa.abs().clamp(min=EPS)


def geometry_name(kappa: torch.Tensor) -> str:
    """Human-readable geometry type based on sign of κ."""
    k = float(kappa.item())
    if k < -KAPPA_EPS:
        return "hyperbolic"
    if k >  KAPPA_EPS:
        return "spherical"
    return "flat"


def is_hyperbolic(kappa: torch.Tensor) -> bool:
    return float(kappa.item()) < -KAPPA_EPS


def is_spherical(kappa: torch.Tensor) -> bool:
    return float(kappa.item()) > KAPPA_EPS


def _curved_weight(kappa: torch.Tensor) -> torch.Tensor:
    """Smooth scalar weight in [0, 1) for blending flat and curved maps.

    Returns tanh(|κ| / GATE_SOFTNESS).  Equals 0 when κ=0 (fully flat) and
    approaches 1 for |κ| >> GATE_SOFTNESS (fully curved).  Differentiable
    everywhere — gradients flow back into κ so the optimiser can push a
    stratum out of the flat regime without hitting a gradient dead zone.
    """
    return torch.tanh(kappa.abs() / GATE_SOFTNESS)


# ── Hyperbolic geometry — Poincaré ball model ──────────────────────────────

def _poincare_clip(x: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    """Project points strictly inside the Poincaré ball of radius 1/√K."""
    max_r = (1.0 / K.sqrt()) * (1.0 - EPS)
    nrm   = x.norm(dim=-1, keepdim=True).clamp(min=EPS)
    scale = (max_r / nrm).clamp(max=1.0).to(x.dtype)
    return x * scale


def _mobius_add(x: torch.Tensor, y: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    """Möbius addition x ⊕_K y in the Poincaré ball."""
    x2  = (x * x).sum(dim=-1, keepdim=True)
    y2  = (y * y).sum(dim=-1, keepdim=True)
    xy  = (x * y).sum(dim=-1, keepdim=True)
    num = (1.0 + 2.0 * K * xy + K * y2) * x + (1.0 - K * x2) * y
    den = (1.0 + 2.0 * K * xy + K**2 * x2 * y2).clamp(min=EPS)
    return num / den


def hyp_exp_origin(v: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    """Exponential map at origin: ℝ^d → Poincaré ball(K).
    exp_0(v) = tanh(√K ‖v‖/2) · v / (√K ‖v‖)
    """
    sqK = K.sqrt()
    nrm = v.norm(dim=-1, keepdim=True).clamp(min=EPS)
    return torch.tanh(sqK * nrm * 0.5) * v / (sqK * nrm)


def hyp_log_origin(x: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    """Logarithmic map at origin: Poincaré ball(K) → ℝ^d.
    log_0(x) = (2/√K) arctanh(√K ‖x‖) · x / ‖x‖
    """
    sqK   = K.sqrt()
    max_r = (1.0 / sqK) * (1.0 - EPS)
    nrm   = x.norm(dim=-1, keepdim=True).clamp(min=EPS)
    nrm_c = nrm.clamp(max=max_r)
    arg   = (sqK * nrm_c).to(torch.float32).clamp(max=1.0 - EPS)
    return (2.0 / sqK) * torch.arctanh(arg) * (x / nrm)


def hyp_dist(x: torch.Tensor, y: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    """Geodesic distance in Poincaré ball(K):
    d(x, y) = (2/√K) arctanh(√K ‖−x ⊕_K y‖)
    """
    sqK  = K.sqrt()
    diff = _mobius_add(-x, y, K)
    nrm  = diff.norm(dim=-1).clamp(max=(1.0 / sqK) * (1.0 - EPS))
    return (2.0 / sqK) * torch.arctanh((sqK * nrm).to(torch.float32))


# ── Spherical geometry — hypersphere model ─────────────────────────────────

def sph_exp_origin(v: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    """Exponential map at north pole of sphere(K), radius r = 1/√K.
    exp_0(v) = r · sin(‖v‖/r) · v / ‖v‖
    """
    r   = 1.0 / K.sqrt()
    nrm = v.norm(dim=-1, keepdim=True).clamp(min=EPS)
    ang = (nrm / r).clamp(max=torch.pi - EPS)
    return r * torch.sin(ang) * v / nrm


def sph_log_origin(x: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    """Logarithmic map from sphere(K) back to tangent space at north pole.
    log_0(x) = r · arcsin(‖x‖/r) · x / ‖x‖
    """
    r     = 1.0 / K.sqrt()
    nrm   = x.norm(dim=-1, keepdim=True).clamp(min=EPS)
    nrm_c = nrm.clamp(max=r * (1.0 - EPS))
    ang   = torch.asin((nrm_c / r).to(torch.float32).clamp(-1.0 + EPS, 1.0 - EPS))
    return r * ang * (x / nrm)


def sph_dist(x: torch.Tensor, y: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    """Geodesic distance on sphere(K):
    d(x, y) = r · arccos(x · y / r²)   where r = 1/√K.
    """
    r   = 1.0 / K.sqrt()
    cos = ((x * y).sum(dim=-1) / (r * r)).clamp(-1.0 + EPS, 1.0 - EPS)
    return r * torch.acos(cos.to(torch.float32))


# ── Unified interface ──────────────────────────────────────────────────────
#
# Each function uses the same two-level blend:
#
#   1. w = _curved_weight(κ) — smooth [0,1) scalar: how "curved" the geometry is.
#   2. sign selection — torch.where(κ > 0, spherical_result, hyperbolic_result)
#      selects which curved geometry to use.  torch.where is differentiable;
#      both branches are always evaluated so gradients flow through both.
#
# Result: gradients reach κ via both w (magnitude of curvature) and the K
# inside each curved map (radius of the manifold), enabling the optimiser to
# continuously steer a stratum toward hyperbolic, flat, or spherical geometry.

def exp_map_origin(v: torch.Tensor, kappa: torch.Tensor) -> torch.Tensor:
    """Map a tangent vector at the origin to the manifold of curvature κ.

    Smooth blend: (1−w)·v  +  w·curved_map(v),
    where w = tanh(|κ| / GATE_SOFTNESS) and curved_map is sph or hyp by sign(κ).
    """
    K = _abs_K(kappa)
    w = _curved_weight(kappa)
    curved = torch.where(kappa > 0, sph_exp_origin(v, K), hyp_exp_origin(v, K))
    return (1.0 - w) * v + w * curved


def log_map_origin(x: torch.Tensor, kappa: torch.Tensor) -> torch.Tensor:
    """Map a manifold point back to the tangent space at the origin.

    Smooth blend: (1−w)·x  +  w·curved_log(x).
    """
    K = _abs_K(kappa)
    w = _curved_weight(kappa)
    curved = torch.where(kappa > 0, sph_log_origin(x, K), hyp_log_origin(x, K))
    return (1.0 - w) * x + w * curved


def clip_to_manifold(x: torch.Tensor, kappa: torch.Tensor) -> torch.Tensor:
    """Project points onto the appropriate manifold for numerical stability.

    Smooth blend between unconstrained (flat) and clipped (curved) points.
    """
    K   = _abs_K(kappa)
    w   = _curved_weight(kappa)
    r   = 1.0 / K.sqrt()
    nrm = x.norm(dim=-1, keepdim=True).clamp(min=EPS)
    # spherical: project onto sphere of radius r
    # hyperbolic: clip to interior of Poincaré ball of radius 1/√K
    clipped = torch.where(kappa > 0, x * (r / nrm), _poincare_clip(x, K))
    return (1.0 - w) * x + w * clipped


def tangent_proj(
    x:      torch.Tensor,  # (N, d) — manifold points
    weight: torch.Tensor,  # (d,)   — direction in tangent space
    kappa:  torch.Tensor,
) -> torch.Tensor:         # (N,)   — scalar lens values
    """Project manifold points to a scalar via log_0 + inner product.

    Maps each point to its tangent-space representation, then projects
    along the learnable direction `weight`.  The tangent space is isometric
    to ℝ^d for all three geometry types, so the lens is always well-defined.
    """
    x_tan = log_map_origin(x, kappa)                          # (N, d) in ℝ^d
    w_hat = F.normalize(weight.float().unsqueeze(0), dim=-1)  # (1, d)
    return (x_tan.float() @ w_hat.T).squeeze(-1)              # (N,)


def dist_to_protos(
    x:      torch.Tensor,   # (N, d) — query points on the manifold
    protos: torch.Tensor,   # (P, d) — prototype points on the manifold
    kappa:  torch.Tensor,
) -> torch.Tensor:          # (N, P) — geodesic distances
    """Geodesic distance from every query point to every prototype.

    Uses the same smooth blend as exp/log_map_origin:
        (1−w)·euclidean  +  w·curved_dist
    where w = _curved_weight(κ) and curved_dist is hyperbolic or spherical
    by sign(κ).  All computations in float32 for numerical stability.
    """
    N, d = x.shape
    P    = protos.shape[0]
    K    = _abs_K(kappa)
    w    = _curved_weight(kappa)

    xi = x.float().unsqueeze(1).expand(N, P, d)      # (N, P, d)
    pi = protos.float().unsqueeze(0).expand(N, P, d)  # (N, P, d)

    d_flat = (xi - pi).norm(dim=-1)                   # (N, P) Euclidean

    # Hyperbolic (Poincaré ball) distance — (N, P)
    xf   = xi.reshape(-1, d)
    pf   = pi.reshape(-1, d)
    sqK  = K.sqrt()
    diff = _mobius_add(-xf, pf, K)
    nrm  = diff.norm(dim=-1).clamp(max=(1.0 / sqK) * (1.0 - EPS))
    d_hyp = ((2.0 / sqK) * torch.arctanh((sqK * nrm).clamp(max=1.0 - EPS))).reshape(N, P)

    # Spherical distance — (N, P)
    r     = 1.0 / K.sqrt()
    cos   = ((xi * pi).sum(-1) / (r * r)).clamp(-1.0 + EPS, 1.0 - EPS)
    d_sph = r * torch.acos(cos)

    d_curved = torch.where(kappa > 0, d_sph, d_hyp)  # (N, P) — select by sign(κ)
    return (1.0 - w) * d_flat + w * d_curved
