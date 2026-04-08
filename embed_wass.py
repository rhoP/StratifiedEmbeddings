"""
embed_wass.py — Pressure-lens Wasserstein shape embedding.

Learns a Riemannian embedding of airfoil shapes such that pairwise geodesic
distances approximate the Wasserstein-1 distance between surface pressure
(Cp) distributions.  Pressure acts as the aerodynamic lens: two shapes are
"nearby" in embedding space if they produce similar chord-wise loading profiles.

Model
-----
ShapeEncoder (GeometricConvAutoencoder backbone; shape channels only):
  per-airfoil patches (P, 5, H, W)  →  mean-pooled embedding  (1, d)

A single learnable curvature κ ∈ (-K_max, K_max) parameterises a constant-
curvature Riemannian metric on R^d:
  κ < 0  →  hyperbolic:  d = (2/√|κ|) · arctanh(√|κ| · ‖e_i−e_j‖/2)
  κ > 0  →  spherical:   d = (2/√κ)   · arctan (√κ   · ‖e_i−e_j‖/2)
  κ = 0  →  Euclidean:   d = ‖e_i−e_j‖

Wasserstein-1 target
--------------------
W₁(Cp_i, Cp_j) = mean_t |sort(Cp_i)[t] − sort(Cp_j)[t]|

Treating each chord-wise Cp profile as a uniform discrete measure over its
N_CHORD values.  The A×A pairwise matrix is built once (vectorised, O(A²·N))
before training.

Training phases
---------------
1. Warmup  — encoder + linear head, MSE on mean surface Cp.
             Gives the encoder aerodynamically meaningful initialisation.
2. Metric  — all-pairs metric regression:
               L = MSE( D_geo[i,j], Ŵ[i,j] )  over lower triangle
             plus a spread regulariser to prevent embedding collapse.
             Dataset order is frozen (no shuffle) so positions match the W matrix.

Outputs (--out_dir)
-------------------
  checkpoint.pt
  embedding_scatter.png    — PCA of embeddings; coloured by Cp / AoA / shape PC1
  distance_correlation.png — d_geo vs W₁ scatter for all airfoil pairs
  kappa_evolution.png      — κ over phase-2 epochs
  metric_loss.png          — phase-2 training loss

Usage
-----
  python embed_wass.py
  python embed_wass.py --embed_dim 32 --warmup_epochs 20 --metric_epochs 80
  python embed_wass.py --reynolds 3e6 --reynolds_tol 0.1   # filter by Re
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.datasets import AirfRANS
from torch_geometric.data import Data

# ── Shared data pipeline ───────────────────────────────────────────────────────
from run import (
    NEAR_FIELD_BOUNDS,
    GRID_H, GRID_W,
    N_PATCHES,
    GraphGrid,
    AirfRANSNearFieldDataset,
    compute_normalisation,
    collect_shape_vectors,
)

# ── Shape encoder + helpers from run_airfrans ──────────────────────────────────
from run_airfrans import (
    ShapeEncoder,
    _BASE_CH,
    filter_dataset_indices,
    get_airfoil_cp,
    compute_cp_profile,
)

# ── DQE warmup head ────────────────────────────────────────────────────────────
from StratifiedEmbedding import WarmupRegressor


# ── Riemannian distance ────────────────────────────────────────────────────────


def geodesic_dist(
    x:     torch.Tensor,
    y:     torch.Tensor,
    kappa: torch.Tensor,
) -> torch.Tensor:
    """Geodesic distance for constant curvature κ (tangent-space model).

    Embeddings are stored as vectors in R^d; κ modifies only the distance
    formula.  For small |κ|, all three branches agree to first order.

    Parameters
    ----------
    x, y  : (..., d)
    kappa : scalar — learnable curvature
    Returns (...,) non-negative distances.
    """
    chord = torch.norm(x - y, dim=-1).clamp(min=1e-8)

    abs_k  = kappa.abs()
    sqrt_k = abs_k.sqrt().clamp(min=1e-8)
    # scaled half-chord; clamp keeps arctanh/arctan arguments safe
    arg = (sqrt_k * chord * 0.5).clamp(max=1.0 - 1e-6)

    hyp = (2.0 / sqrt_k) * torch.arctanh(arg)   # κ < 0
    sph = (2.0 / sqrt_k) * torch.arctan(arg)     # κ > 0

    return torch.where(kappa < -1e-5, hyp, torch.where(kappa > 1e-5, sph, chord))


# ── Model ──────────────────────────────────────────────────────────────────────


class WassersteinEmbedder(nn.Module):
    """Shape encoder + learnable Riemannian curvature for metric learning.

    Call ``embed(patches)`` to obtain a (1, embed_dim) airfoil embedding,
    and ``pairwise_geodesic(E)`` for an (A, A) distance matrix.

    Parameters
    ----------
    embed_dim     : embedding dimension
    base_channels : GeometricConvAutoencoder base filter count
    K_max         : curvature clamped to (−K_max, K_max) via tanh
    """

    def __init__(
        self,
        embed_dim:     int   = 64,
        base_channels: int   = _BASE_CH,
        dropout:       float = 0.1,
        K_max:         float = 2.0,
    ) -> None:
        super().__init__()
        self.encoder     = ShapeEncoder(embed_dim, base_channels, dropout)
        self.warmup_head = WarmupRegressor(embed_dim, n_targets=1)
        self._log_kappa  = nn.Parameter(torch.zeros(1))
        self._K_max      = K_max

    @property
    def kappa(self) -> torch.Tensor:
        """Learnable curvature κ ∈ (−K_max, K_max)."""
        return torch.tanh(self._log_kappa) * self._K_max

    def embed(self, patches: torch.Tensor) -> torch.Tensor:
        """(P, 7, H, W) → (1, embed_dim) — mean-pooled airfoil embedding."""
        return self.encoder(patches).mean(0, keepdim=True)

    def warmup_forward(self, patches: torch.Tensor) -> torch.Tensor:
        return self.warmup_head(self.embed(patches))   # (1, 1) Cp prediction

    def pairwise_geodesic(self, E: torch.Tensor) -> torch.Tensor:
        """(A, d) → (A, A) pairwise geodesic distance matrix."""
        return geodesic_dist(
            E.unsqueeze(1),   # (A, 1, d)
            E.unsqueeze(0),   # (1, A, d)
            self.kappa,
        )

    def set_phase(self, phase: int) -> None:
        for p in self.parameters():
            p.requires_grad_(False)
        if phase == 1:
            for p in self.encoder.parameters():     p.requires_grad_(True)
            for p in self.warmup_head.parameters(): p.requires_grad_(True)
        elif phase == 2:
            for p in self.encoder.parameters():     p.requires_grad_(True)
            self._log_kappa.requires_grad_(True)


# ── Wasserstein target matrix ──────────────────────────────────────────────────


def build_wasserstein_matrix(profiles: np.ndarray) -> np.ndarray:
    """Pairwise W₁ between chord-wise Cp profiles.

    Discrete 1-D Wasserstein treating each profile as a uniform empirical
    measure over its N_CHORD values:
      W₁(p, q) = (1/N) ∑ |sort(p) - sort(q)|

    Fully vectorised: O(A²·N) time, O(A²·N) peak memory.

    Parameters
    ----------
    profiles : (A, N_CHORD) normalised chord-wise Cp

    Returns
    -------
    W : (A, A) symmetric non-negative matrix
    """
    ps   = np.sort(profiles, axis=1)                          # (A, N)
    diff = ps[:, np.newaxis, :] - ps[np.newaxis, :, :]        # (A, A, N)
    return np.abs(diff).mean(axis=2).astype(np.float32)       # (A, A)


# ── Data helpers ───────────────────────────────────────────────────────────────


def collect_cp_profiles_ordered(
    raw_ds:  AirfRANS,
    dataset: AirfRANSNearFieldDataset,
    p_mean:  float,
    p_std:   float,
) -> np.ndarray:
    """(A, N_CHORD) chord-wise Cp profiles in current dataset iteration order.

    Mirrors ``run_airfrans.collect_cp_profiles`` but uses ``P_IDX`` from
    ``run.py``'s constants directly via ``raw.y[:, 2]`` (pressure is target
    column 2 in AirfRANS).
    """
    P_IDX_LOCAL = 2   # pressure column in raw.y
    profiles: list[np.ndarray] = []
    for i in range(len(dataset)):
        raw_idx = dataset._indices[dataset._order[i]]
        raw: Data = raw_ds[raw_idx]  # type: ignore[assignment]
        pos  = cast(torch.Tensor, raw.pos).float().numpy()
        surf = (
            cast(torch.Tensor, raw.surf).bool().numpy()
            if hasattr(raw, "surf") and raw.surf is not None
            else np.zeros(pos.shape[0], dtype=bool)
        )
        p_norm = (cast(torch.Tensor, raw.y).float().numpy()[:, P_IDX_LOCAL] - p_mean) / p_std
        profiles.append(compute_cp_profile(pos[surf], p_norm[surf]))
    return np.stack(profiles)


def collect_aoas(
    raw_ds:   AirfRANS,
    datasets: list[AirfRANSNearFieldDataset],
) -> np.ndarray:
    """(A,) angle of attack (degrees) for all airfoils, in iteration order."""
    aoas: list[float] = []
    for ds in datasets:
        for i in range(len(ds)):
            raw_idx = ds._indices[ds._order[i]]
            g: Data = raw_ds[raw_idx]  # type: ignore[assignment]
            ux = float(cast(torch.Tensor, g.x)[0, 0])
            uy = float(cast(torch.Tensor, g.x)[0, 1])
            aoas.append(math.degrees(math.atan2(uy, ux)))
    return np.array(aoas, dtype=np.float32)


@torch.no_grad()
def collect_embeddings(
    model:    WassersteinEmbedder,
    datasets: list[AirfRANSNearFieldDataset],
    device:   torch.device,
) -> tuple[torch.Tensor, np.ndarray]:
    """(A, d) embeddings and (A,) mean-Cp for all airfoils."""
    model.eval()
    embs, cps = [], []
    for ds in datasets:
        for gg in ds:
            embs.append(model.embed(gg.patches.to(device)).cpu())
            cps.append(get_airfoil_cp(gg).item())
    return torch.cat(embs, dim=0), np.array(cps, dtype=np.float32)


# ── Spread regularisation ──────────────────────────────────────────────────────


def spread_loss(E: torch.Tensor, margin: float) -> torch.Tensor:
    """Push embeddings apart: hinge on pairs closer than `margin`."""
    D   = torch.cdist(E, E)
    off = ~torch.eye(E.shape[0], dtype=torch.bool, device=E.device)
    return F.relu(margin - D[off]).mean()


# ── Visualisations ─────────────────────────────────────────────────────────────


def plot_embedding_scatter(
    embeddings: np.ndarray,   # (A, d)
    cps:        np.ndarray,   # (A,)  normalised
    all_shape:  np.ndarray,   # (A, 2*N_CHORD) geometric shape vectors
    aoas:       np.ndarray,   # (A,) degrees
    p_mean:     float,
    p_std:      float,
    out_path:   Path,
) -> None:
    """PCA of embeddings coloured by mean Cp, AoA, and geometric shape PC1."""
    E_c = embeddings - embeddings.mean(0)
    _, _, Vt = np.linalg.svd(E_c, full_matrices=False)
    coords = E_c @ Vt[:2].T                           # (A, 2)

    S_c = all_shape - all_shape.mean(0)
    _, _, Vt_s = np.linalg.svd(S_c, full_matrices=False)
    shape_pc1 = S_c @ Vt_s[0]                         # (A,)

    cp_phys = cps * p_std + p_mean

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    triplets = [
        (cp_phys,  "Mean surface Cp",          "RdBu_r"),
        (aoas,     "Angle of Attack (°)",       "coolwarm"),
        (shape_pc1,"Geometric shape PC 1",      "viridis"),
    ]
    for ax, (vals, label, cmap) in zip(axes, triplets):
        sc = ax.scatter(
            coords[:, 0], coords[:, 1],
            c=vals, cmap=cmap, s=45, edgecolors="k", linewidths=0.3, alpha=0.88,
        )
        fig.colorbar(sc, ax=ax, label=label, fraction=0.046, pad=0.04)
        ax.set_xlabel("Embedding PC 1")
        ax.set_ylabel("Embedding PC 2")
        ax.set_title(f"Coloured by {label}")
        ax.grid(lw=0.3, alpha=0.4)
        r, _ = pearsonr(coords[:, 0], vals)
        ax.annotate(
            f"r(PC1) = {r:+.3f}",
            xy=(0.04, 0.96), xycoords="axes fraction",
            va="top", ha="left", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#999", alpha=0.85),
        )

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


def plot_distance_correlation(
    D_geo:    np.ndarray,   # (A, A) geodesic
    W_target: np.ndarray,   # (A, A) normalised W₁
    out_path: Path,
) -> None:
    """Scatter of geodesic distance vs W₁ target for all unique pairs."""
    A   = D_geo.shape[0]
    idx = np.tril_indices(A, k=-1)
    d_v = D_geo[idx]
    w_v = W_target[idx]

    r_p, _ = pearsonr(d_v, w_v)
    r_s, _ = spearmanr(d_v, w_v)

    # Hex-bin density for the (potentially large) number of pairs
    fig, ax = plt.subplots(figsize=(6, 6))
    hb = ax.hexbin(w_v, d_v, gridsize=50, cmap="YlOrRd", mincnt=1)
    fig.colorbar(hb, ax=ax, label="pair count")
    # Overlay a reference line (least-squares fit)
    m, b = np.polyfit(w_v, d_v, 1)
    xlim = np.array([w_v.min(), w_v.max()])
    ax.plot(xlim, m * xlim + b, "steelblue", lw=1.5, ls="--", label="linear fit")
    ax.set_xlabel("W₁(Cp_i, Cp_j)  [target, normalised]")
    ax.set_ylabel("Geodesic distance  [learned]")
    ax.set_title("Embedding vs Wasserstein target")
    ax.annotate(
        f"Pearson r  = {r_p:.3f}\nSpearman ρ = {r_s:.3f}",
        xy=(0.04, 0.96), xycoords="axes fraction",
        va="top", ha="left", fontsize=10,
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#999", alpha=0.9),
    )
    ax.legend(fontsize=9)
    ax.grid(lw=0.4, alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


def plot_kappa_evolution(kappa_hist: list[float], out_path: Path) -> None:
    """Learnable curvature κ over phase-2 epochs."""
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(kappa_hist, lw=2, color="steelblue")
    ax.axhline(0.0, color="k", lw=0.8, ls="--", alpha=0.6, label="κ=0  (Euclidean)")
    ax.fill_between(range(len(kappa_hist)), kappa_hist, 0,
                    where=[k < 0 for k in kappa_hist],
                    color="royalblue", alpha=0.15, label="hyperbolic")
    ax.fill_between(range(len(kappa_hist)), kappa_hist, 0,
                    where=[k > 0 for k in kappa_hist],
                    color="tomato", alpha=0.15, label="spherical")
    ax.set_xlabel("Epoch (phase 2)")
    ax.set_ylabel("κ")
    ax.set_title("Curvature evolution")
    ax.legend(fontsize=9)
    ax.grid(lw=0.4, alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


def plot_metric_loss(loss_hist: list[float], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(loss_hist, lw=2, color="coral")
    ax.set_xlabel("Epoch (phase 2)")
    ax.set_ylabel("Metric loss (MSE)")
    ax.set_title("Phase-2 metric regression loss")
    ax.grid(lw=0.4, alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


# ── Main training loop ─────────────────────────────────────────────────────────


def train(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    )
    print(f"Device: {device}")

    NEAR_FIELD_BOUNDS["y_min"] = -args.near_field_y
    NEAR_FIELD_BOUNDS["y_max"] =  args.near_field_y

    # ── Data ──────────────────────────────────────────────────────────────────
    print(f"Loading AirfRANS (task={args.task}) …")
    raw_train = AirfRANS(root=args.data_root, task=args.task, train=True)
    raw_test  = AirfRANS(root=args.data_root, task=args.task, train=False)

    # Optional flow-condition filter (same logic as run_airfrans.py)
    re_filter = args.reynolds if args.reynolds > 0 else None
    ma_filter = args.mach     if args.mach     > 0 else None
    if re_filter is not None or ma_filter is not None:
        fparts: list[str] = []
        if re_filter is not None:
            fparts.append(f"Re={re_filter:.3e} ±{args.reynolds_tol*100:.0f}%")
        if ma_filter is not None:
            fparts.append(f"Ma={ma_filter:.4f} ±{args.mach_tol*100:.0f}%")
        print(f"  Filtering: {', '.join(fparts)}")
        train_pool = filter_dataset_indices(
            raw_train, re_filter, ma_filter, args.reynolds_tol, args.mach_tol
        )
        test_pool = filter_dataset_indices(
            raw_test, re_filter, ma_filter, args.reynolds_tol, args.mach_tol
        )
        if len(train_pool) == 0:
            raise ValueError(
                "Filter matched 0 training samples. "
                "Widen --reynolds_tol / --mach_tol or adjust target values."
            )
        print(
            f"  Kept {len(train_pool)}/{len(raw_train)} train  "
            f"{len(test_pool)}/{len(raw_test)} test"
        )
    else:
        train_pool = list(range(len(raw_train)))
        test_pool  = list(range(len(raw_test)))

    val_size  = max(1, int(len(train_pool) * 0.1))
    train_idx = train_pool[: len(train_pool) - val_size]
    val_idx   = train_pool[len(train_pool) - val_size :]

    print("  Computing normalisation stats …")
    x_mean, x_std, p_mean_t, p_std_t = compute_normalisation(raw_train, train_idx)
    norm_stats = (x_mean, x_std, p_mean_t, p_std_t)
    p_mean_v = float(p_mean_t)
    p_std_v  = float(p_std_t)

    shared_cache: dict[int, GraphGrid] = {}
    train_ds = AirfRANSNearFieldDataset(
        args.data_root, "train", task=args.task,
        norm_stats=norm_stats, indices=train_idx,
        _shared_raw=raw_train, _shared_cache=shared_cache,
    )
    val_ds = AirfRANSNearFieldDataset(
        args.data_root, "train", task=args.task,
        norm_stats=norm_stats, indices=val_idx,
        _shared_raw=raw_train, _shared_cache=shared_cache,
    )
    test_ds = AirfRANSNearFieldDataset(
        args.data_root, "test", task=args.task,
        norm_stats=norm_stats, indices=test_pool,
        _shared_raw=raw_test,
    )
    n_tr = len(train_ds)
    print(
        f"  {n_tr} train / {len(val_ds)} val / {len(test_ds)} test\n"
        f"  Grid {GRID_H}×{GRID_W}  Patches/airfoil: {N_PATCHES}"
    )

    # ── Wasserstein target matrix (built in default iteration order) ───────────
    # Important: train_ds._order must be [0, 1, ..., n_tr-1] here so that
    # W_train[i,j] corresponds to airfoil at position i vs position j.
    # Phase 2 also iterates without shuffling so the correspondence is preserved.
    print("\n=== Precomputing Wasserstein target matrix ===")
    print(f"  Collecting Cp profiles for {n_tr} training airfoils …")
    prof_tr = collect_cp_profiles_ordered(raw_train, train_ds, p_mean_v, p_std_v)

    print(f"  Building {n_tr}×{n_tr} W₁ matrix (vectorised) …")
    W_train_np = build_wasserstein_matrix(prof_tr)   # (n_tr, n_tr)

    # Normalise to [0, 1] so the metric regression loss has a consistent scale
    w_scale = float(W_train_np.max()) or 1.0
    W_train_t = torch.from_numpy(W_train_np / w_scale).to(device)
    print(f"  W₁ range: [{W_train_np.min():.4f}, {W_train_np.max():.4f}]  scale={w_scale:.4f}")

    # ── Model ──────────────────────────────────────────────────────────────────
    model = WassersteinEmbedder(
        embed_dim     = args.embed_dim,
        base_channels = args.base_channels,
        dropout       = args.dropout,
        K_max         = args.K_max,
    ).to(device)

    # ── Phase 1: Warmup ────────────────────────────────────────────────────────
    print(f"\n=== Phase 1: Warmup ({args.warmup_epochs} epochs) ===")
    model.set_phase(1)
    opt1   = AdamW([p for p in model.parameters() if p.requires_grad],
                   lr=args.lr, weight_decay=1e-4)
    sched1 = CosineAnnealingLR(opt1, T_max=max(1, args.warmup_epochs))

    for epoch in tqdm(range(args.warmup_epochs), desc="warmup", unit="ep"):
        model.train()
        ep_loss = 0.0
        train_ds.shuffle()
        for gg in train_ds:
            patches_d = gg.patches.to(device)
            target_d  = get_airfoil_cp(gg).to(device)
            opt1.zero_grad()
            pred  = model.warmup_forward(patches_d)
            loss  = F.mse_loss(pred, target_d)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt1.step()
            ep_loss += loss.item()
            del patches_d, target_d, pred
        sched1.step()
        if (epoch + 1) % max(1, args.warmup_epochs // 5) == 0:
            print(f"  [warmup] epoch {epoch+1:3d}  loss={ep_loss/n_tr:.4f}")

    # ── Phase 2: Metric regression ─────────────────────────────────────────────
    print(f"\n=== Phase 2: Metric regression ({args.metric_epochs} epochs) ===")
    model.set_phase(2)

    # Reset iteration order to default so positions match W_train_t indices
    train_ds._order = list(range(n_tr))

    opt2   = AdamW([p for p in model.parameters() if p.requires_grad],
                   lr=args.lr, weight_decay=1e-4)
    sched2 = CosineAnnealingLR(opt2, T_max=max(1, args.metric_epochs))

    # Lower-triangle mask: (n_tr, n_tr) — excludes diagonal and upper triangle
    mask_tril = torch.tril(
        torch.ones(n_tr, n_tr, dtype=torch.bool, device=device), diagonal=-1
    )

    loss_hist:  list[float] = []
    kappa_hist: list[float] = []
    best_loss = math.inf

    epoch_bar = tqdm(range(args.metric_epochs), desc="metric", unit="ep")
    for epoch in epoch_bar:
        model.train()

        # Collect all train embeddings with gradients.
        # No shuffle in phase 2 — position i must correspond to W_train_t row i.
        embs_list: list[torch.Tensor] = []
        for gg in train_ds:
            embs_list.append(model.embed(gg.patches.to(device)))   # (1, d)
        E = torch.cat(embs_list, dim=0)   # (n_tr, d), fully differentiable

        # Pairwise geodesic distances
        D_geo = model.pairwise_geodesic(E)   # (n_tr, n_tr)

        # Metric regression on unique pairs
        l_metric = F.mse_loss(D_geo[mask_tril], W_train_t[mask_tril])

        # Spread regularisation: push embeddings apart to prevent collapse
        l_spread = spread_loss(E.detach(), args.spread_margin)

        loss = l_metric + args.spread_weight * l_spread
        opt2.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt2.step()
        sched2.step()

        kappa_v = model.kappa.item()
        loss_hist.append(l_metric.item())
        kappa_hist.append(kappa_v)
        epoch_bar.set_postfix(
            metric=f"{l_metric.item():.5f}",
            kappa=f"{kappa_v:+.3f}",
        )

        if (epoch + 1) % max(1, args.metric_epochs // 10) == 0:
            if l_metric.item() < best_loss:
                best_loss = l_metric.item()
                torch.save(
                    {
                        "model": model.state_dict(),
                        "epoch": epoch + 1,
                        "kappa": kappa_v,
                        "w_scale": w_scale,
                        "norm_stats": {
                            "x_mean": x_mean.cpu(), "x_std": x_std.cpu(),
                            "p_mean": p_mean_t.cpu(), "p_std": p_std_t.cpu(),
                        },
                    },
                    out_dir / "checkpoint.pt",
                )

    kappa_final = model.kappa.item()
    geom = "hyperbolic" if kappa_final < 0 else ("spherical" if kappa_final > 0 else "Euclidean")
    print(f"\nFinal κ = {kappa_final:+.4f}  ({geom})")

    # ── Representations for visualisation ─────────────────────────────────────
    print("\n=== Collecting representations ===")
    torch.cuda.empty_cache()

    all_embs, all_cps = collect_embeddings(model, [train_ds, val_ds, test_ds], device)
    all_embs_np = all_embs.numpy().astype(np.float64)

    # AoA: only from raw_train for train/val; raw_test for test
    train_val_aoas = collect_aoas(raw_train, [train_ds, val_ds])
    test_aoas      = collect_aoas(raw_test,  [test_ds])
    all_aoas = np.concatenate([train_val_aoas, test_aoas])

    # Geometric shape vectors (thickness + camber profiles)
    shape_tr = collect_shape_vectors(raw_train, train_ds)
    shape_va = collect_shape_vectors(raw_train, val_ds)
    shape_te = collect_shape_vectors(raw_test,  test_ds)
    all_shape = np.concatenate([shape_tr, shape_va, shape_te])

    # Pairwise distances for full set (used in distance-correlation plot)
    print("  Building full W₁ matrix …")
    prof_va = collect_cp_profiles_ordered(raw_train, val_ds,  p_mean_v, p_std_v)
    prof_te = collect_cp_profiles_ordered(raw_test,  test_ds, p_mean_v, p_std_v)
    all_profiles  = np.concatenate([prof_tr, prof_va, prof_te])
    W_all_np      = build_wasserstein_matrix(all_profiles) / w_scale

    with torch.no_grad():
        D_geo_all = model.pairwise_geodesic(all_embs.to(device)).cpu().numpy()

    # ── Final metrics ──────────────────────────────────────────────────────────
    A_all = len(all_embs)
    tri   = np.tril_indices(A_all, k=-1)
    r_p, _ = pearsonr(D_geo_all[tri], W_all_np[tri])
    r_s, _ = spearmanr(D_geo_all[tri], W_all_np[tri])
    print(f"\nDistance correlation (all {A_all} airfoils, {len(tri[0])} pairs):")
    print(f"  Pearson r  = {r_p:.4f}")
    print(f"  Spearman ρ = {r_s:.4f}")

    metrics = {
        "pearson_r":        r_p,
        "spearman_rho":     r_s,
        "kappa":            kappa_final,
        "geometry":         geom,
        "best_metric_loss": best_loss,
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # ── Visualisations ─────────────────────────────────────────────────────────
    print("\n=== Saving visualisations ===")
    plot_embedding_scatter(
        all_embs_np, all_cps, all_shape, all_aoas,
        p_mean_v, p_std_v,
        out_dir / "embedding_scatter.png",
    )
    plot_distance_correlation(
        D_geo_all, W_all_np,
        out_dir / "distance_correlation.png",
    )
    plot_kappa_evolution(kappa_hist, out_dir / "kappa_evolution.png")
    plot_metric_loss(loss_hist, out_dir / "metric_loss.png")

    print(f"\nAll outputs saved to {out_dir}")


# ── CLI ────────────────────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Pressure-lens Wasserstein shape embedding on a Riemannian manifold",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Data
    p.add_argument("--data_root",    default="../data/AirfRANS")
    p.add_argument("--task",         default="scarce", choices=["scarce", "full"])
    p.add_argument("--near_field_y", type=float, default=0.6,
                   help="Must match the value used in run.py")
    # Model
    p.add_argument("--embed_dim",     type=int,   default=64)
    p.add_argument("--base_channels", type=int,   default=32,
                   help="GeometricConvAutoencoder base filter count")
    p.add_argument("--dropout",       type=float, default=0.1)
    p.add_argument("--K_max",         type=float, default=2.0,
                   help="Curvature clamped to (−K_max, K_max)")
    # Training
    p.add_argument("--warmup_epochs", type=int,   default=30)
    p.add_argument("--metric_epochs", type=int,   default=100)
    p.add_argument("--lr",            type=float, default=3e-4)
    # Loss
    p.add_argument("--spread_weight", type=float, default=0.1,
                   help="Weight for the embedding spread regulariser")
    p.add_argument("--spread_margin", type=float, default=0.5,
                   help="Minimum pairwise distance encouraged by the spread loss")
    # Flow-condition filter (0 = disabled)
    p.add_argument("--reynolds",      type=float, default=0,
                   help="Target Re; 0 = no filter  (e.g. 3e6)")
    p.add_argument("--reynolds_tol",  type=float, default=0.05,
                   help="Fractional tolerance for Re filter  (default ±5%%)")
    p.add_argument("--mach",          type=float, default=0,
                   help="Target Ma; 0 = no filter  (e.g. 0.15)")
    p.add_argument("--mach_tol",      type=float, default=0.05,
                   help="Fractional tolerance for Ma filter  (default ±5%%)")
    # Misc
    p.add_argument("--out_dir", default="results/wass_embedding")
    p.add_argument("--seed",    type=int, default=42)
    p.add_argument("--cpu",     action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    t0 = time.time()
    train(args)
    print(f"\nTotal wall time: {(time.time()-t0)/60:.1f} min")
