"""
run_airfrans.py — Shape-Stratified DQE for AirfRANS airfoils.

Objective
---------
Discover K shape families among airfoil geometries, where each stratum is
defined by its characteristic pressure distribution.  Pressure acts as the
ground-truth lens: strata that group airfoils with similar Cp behaviour are
rewarded, so the K strata implicitly learn aerodynamically meaningful shape
classes (thin/thick, high/low camber, leading-edge type, etc.).

Architecture
------------
For each airfoil simulation:
  1. Interpolate the near-field to a 128×128 grid; extract 32×32 patches.
  2. ShapeEncoder (GeometricConvAutoencoder CNN body + AdaptiveAvgPool):
       input  — shape-only channels [x, y, SDF, nx, ny] (inlet velocity excluded)
       output — (P, embed_dim) per-patch embeddings
  3. Mean-pool over patches → (1, embed_dim) per-airfoil embedding.
  4. StratumAssigner → (1, K) soft assignment weights.
  5. K × StratumDQE heads, each predicting mean surface Cp.
  6. Weighted mixture → final Cp prediction.

Training phases
---------------
1. Warmup   — ShapeEncoder + linear head, plain MSE on mean surface Cp.
2. K-means  — cluster pooled embeddings; initialise assigner + DQE prototypes.
3. Assigner — train StratumAssigner to reproduce K-means labels (encoder frozen).
4. Full     — all parameters; loss = MSE + entropy + curvature_diversity + centripetal.

Visualisations (auto-generated after training)
----------------------------------------------
  training_curves.png        — loss components over phase 4
  curvature_evolution.png    — κ per stratum over phase 4
  stratum_scatter.png        — embedding PCA; left=stratum colour, right=Cp colour
  stratum_distributions.png  — violin: Cp distribution per stratum
  stratum_cp_profiles.png    — mean chord-wise Cp profile per stratum (± 1σ band)
  shape_stratum.png          — geometric shape PCA coloured by stratum assignment

Usage
-----
  python run_airfrans.py --out_dir results/shape_strata
  python run_airfrans.py --n_strata 2 --embed_dim 32 --warmup_epochs 5 --full_epochs 20
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
from scipy.stats import pearsonr
from tqdm import tqdm
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.datasets import AirfRANS
from torch_geometric.data import Data

# ── Shared data pipeline from run.py ─────────────────────────────────────────
from run import (
    NEAR_FIELD_BOUNDS,
    GRID_H, GRID_W,
    N_PATCHES, P_IDX, N_CHORD,
    GraphGrid,
    AirfRANSNearFieldDataset,
    compute_normalisation,
    collect_shape_vectors,
)

# ── CNN backbone ──────────────────────────────────────────────────────────────
from models.GeometricCNNAutoencoder import GeometricConvAutoencoder

# ── DQE components ────────────────────────────────────────────────────────────
from StratifiedEmbedding import (
    StratumAssigner,
    StratumDQE,
    WarmupRegressor,
    stratum_entropy_loss,
    curvature_diversity_loss,
    centripetal_loss,
)

# ── Constants ─────────────────────────────────────────────────────────────────
# Channel indices in the 7-channel patch [x, y, u_x_inf, u_y_inf, SDF, nx, ny]
# Shape-only: drop inlet velocity (indices 2, 3)
_FLOW_IDX  = [0, 1, 4]   # x, y, SDF  → "flow" channels for GeometricConvAutoencoder
_GEOM_IDX  = [5, 6]       # nx, ny     → "geometry" channels
_BASE_CH   = 32           # GeometricConvAutoencoder base channels

# Air properties at ~20 °C / sea level (used to compute Re and Ma from U_inf)
_RHO   = 1.2          # kg/m³  — density
_MU    = 1.8e-5       # Pa·s   — dynamic viscosity
_C_SND = 340.0        # m/s    — speed of sound
_CHORD = 1.0          # m      — reference chord length (AirfRANS convention)


# ── Model ─────────────────────────────────────────────────────────────────────


class ShapeEncoder(nn.Module):
    """CNN patch encoder built on the GeometricConvAutoencoder backbone.

    Selects shape-only channels [x, y, SDF, nx, ny] from the full 7-channel
    patches before encoding.  The CNN body is taken directly from
    ``GeometricConvAutoencoder`` (4 stride-2 conv blocks), and an
    ``AdaptiveAvgPool2d(1)`` + linear projection replace the original
    flatten+FC so the encoder is patch-size-agnostic.

    Parameters
    ----------
    embed_dim     : output embedding dimension
    base_channels : base filter count for GeometricConvAutoencoder (default 32;
                    final conv channel count = base_channels * 8)
    dropout       : dropout before projection
    """

    def __init__(
        self,
        embed_dim: int,
        base_channels: int = _BASE_CH,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        # Borrow the CNN body (nn.Sequential) — 4 stride-2 conv blocks,
        # output shape (P, base_channels*8, H/16, W/16).
        _cae = GeometricConvAutoencoder(
            latent_dim=embed_dim,
            flow_channels=len(_FLOW_IDX),
            geom_channels=len(_GEOM_IDX),
            base_channels=base_channels,
        )
        self.cnn     = _cae.encoder            # (P, _SHAPE_CH, H, W) → (P, BC*8, h, w)
        self.pool    = nn.AdaptiveAvgPool2d(1)
        self.proj    = nn.Linear(base_channels * 8, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        """patches : (P, 7, H, W) → (P, embed_dim)"""
        x = torch.cat([
            patches[:, _FLOW_IDX, :, :],   # (P, 3, H, W)
            patches[:, _GEOM_IDX, :, :],   # (P, 2, H, W)
        ], dim=1)                           # (P, 5, H, W)
        x = self.cnn(x)                    # (P, BC*8, h, w)
        x = self.pool(x).flatten(1)        # (P, BC*8)
        return self.proj(self.dropout(x))  # (P, embed_dim)


class ShapeStratifiedDQE(nn.Module):
    """Shape-Stratified DQE: CNN patch encoder + graph pooling + K DQE heads.

    One model instance processes one airfoil simulation at a time.
    ``pool_patches`` encodes all patches and mean-pools them to a single
    (1, embed_dim) airfoil-level embedding.  The ``StratumAssigner`` softly
    routes the airfoil to K strata; each ``StratumDQE`` predicts mean
    surface Cp within that stratum.

    Phases mirror StratifiedDQE conventions:
      1 — ShapeEncoder + warmup_head
      2 — StratumAssigner only (encoder frozen)
      3 — everything except warmup_head
    """

    def __init__(
        self,
        embed_dim:     int,
        n_strata:      int,
        n_intervals:   int,
        n_protos:      int,
        base_channels: int   = _BASE_CH,
        dropout:       float = 0.1,
        K_max:         float = 2.0,
        interval_temp: float = 0.5,
        proto_temp:    float = 1.0,
    ) -> None:
        super().__init__()
        self.n_strata  = n_strata
        self.embed_dim = embed_dim

        self.encoder     = ShapeEncoder(embed_dim, base_channels, dropout)
        self.assigner    = StratumAssigner(embed_dim, n_strata)
        self.dqes        = nn.ModuleList([
            StratumDQE(embed_dim, n_intervals, n_protos, n_targets=1,
                       K_max=K_max, interval_temp=interval_temp,
                       proto_temp=proto_temp)
            for _ in range(n_strata)
        ])
        self.warmup_head = WarmupRegressor(embed_dim, n_targets=1)

    # ── phase gating ──────────────────────────────────────────────────────────

    def set_phase(self, phase: int) -> None:
        for p in self.parameters():
            p.requires_grad_(False)
        if phase == 1:
            for p in self.encoder.parameters():     p.requires_grad_(True)
            for p in self.warmup_head.parameters(): p.requires_grad_(True)
        elif phase == 2:
            for p in self.assigner.parameters():    p.requires_grad_(True)
        elif phase == 3:
            for p in self.encoder.parameters():     p.requires_grad_(True)
            for p in self.assigner.parameters():    p.requires_grad_(True)
            for dqe in self.dqes:
                for p in dqe.parameters():          p.requires_grad_(True)

    # ── forward helpers ───────────────────────────────────────────────────────

    def pool_patches(self, patches: torch.Tensor) -> torch.Tensor:
        """Encode all patches and mean-pool → (1, embed_dim)."""
        return self.encoder(patches).mean(0, keepdim=True)

    def warmup_forward(self, patches: torch.Tensor) -> torch.Tensor:
        """Phase 1: pooled embedding → linear head → (1, 1) Cp prediction."""
        return self.warmup_head(self.pool_patches(patches))

    def forward(
        self, patches: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full forward pass (phase 3).

        Returns
        -------
        pred       : (1, 1)             weighted-mixture Cp prediction
        soft       : (1, K)             soft stratum assignments
        g          : (1, embed_dim)     airfoil embedding
        protos_tan : (K, n_protos, d)   prototype tangent coordinates
        kappas     : (K,)               learnable curvature per stratum
        """
        g    = self.pool_patches(patches)                 # (1, d)
        soft = self.assigner(g)                           # (1, K)

        preds, kappas = [], []
        for dqe in self.dqes:
            assert isinstance(dqe, StratumDQE)
            p, _, _ = dqe(g)
            preds.append(p)
            kappas.append(dqe.kappa)

        preds_stack = torch.stack(preds, dim=1)                   # (1, K, 1)
        pred        = (soft.unsqueeze(-1) * preds_stack).sum(1)   # (1, 1)
        kappas_t    = torch.stack(kappas)                          # (K,)
        protos_tan  = torch.stack(
            [d.protos_tan for d in self.dqes if isinstance(d, StratumDQE)]
        )                                                           # (K, P, d)

        return pred, soft, g, protos_tan, kappas_t

    # ── K-means initialisation ────────────────────────────────────────────────

    @torch.no_grad()
    def init_from_kmeans(
        self,
        centres:    torch.Tensor,   # (K, d)
        labels:     torch.Tensor,   # (A,)
        all_embeds: torch.Tensor,   # (A, d)
    ) -> None:
        self.assigner.init_centroids(centres)
        dqe_list = [d for d in self.dqes if isinstance(d, StratumDQE)]
        for k, dqe in enumerate(dqe_list):
            mask = labels == k
            if mask.sum() == 0:
                continue
            cluster_e = all_embeds[mask]
            P   = dqe.n_protos
            idx = torch.randperm(cluster_e.shape[0])[:P]
            if idx.numel() > 0:
                dqe.protos_tan.data[:idx.numel()].copy_(cluster_e[idx])


# ── Training helpers ──────────────────────────────────────────────────────────


def get_airfoil_cp(gg: GraphGrid) -> torch.Tensor:
    """Return (1, 1) mean surface Cp (normalised) from patch data."""
    smask = gg.patch_surf
    p     = gg.patch_pressure
    val   = p[smask].mean() if smask.any() else p.mean()
    return val.view(1, 1)


@torch.no_grad()
def collect_airfoil_embeddings(
    model:   ShapeStratifiedDQE,
    dataset: AirfRANSNearFieldDataset,
    device:  torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collect per-airfoil (embed, soft_assign, cp) for all graphs in dataset.

    Returns
    -------
    embeds : (A, embed_dim)
    softs  : (A, K)
    cps    : (A,)         normalised mean surface Cp
    """
    model.eval()
    embeds, softs, cps = [], [], []
    for gg in dataset:
        patches_d = gg.patches.to(device)
        _, soft, g, _, _ = model(patches_d)
        embeds.append(g.cpu())
        softs.append(soft.cpu())
        cps.append(get_airfoil_cp(gg).squeeze())
        del patches_d, soft, g
    return (
        torch.cat(embeds, dim=0),
        torch.cat(softs,  dim=0),
        torch.stack(cps),
    )


@torch.no_grad()
def collect_pooled_embeddings(
    model:   ShapeStratifiedDQE,
    dataset: AirfRANSNearFieldDataset,
    device:  torch.device,
) -> torch.Tensor:
    """(A, embed_dim) — used for K-means initialisation after Phase 1."""
    model.eval()
    embs = []
    for gg in dataset:
        g = model.pool_patches(gg.patches.to(device))
        embs.append(g.cpu())
        del g
    return torch.cat(embs, dim=0)


def kmeans_cluster(
    embeds:    torch.Tensor,
    n_clusters: int,
    n_iter:    int  = 100,
    seed:      int  = 42,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Lloyd k-means in Euclidean space.  Returns (centroids (K,d), labels (N,))."""
    torch.manual_seed(seed)
    N = embeds.shape[0]
    centroids = embeds[torch.randperm(N)[:n_clusters]].clone()
    labels    = torch.zeros(N, dtype=torch.long)
    for _ in range(n_iter):
        new_labels = torch.cdist(embeds, centroids).argmin(dim=-1)
        if (new_labels == labels).all():
            break
        labels = new_labels
        for k in range(n_clusters):
            mask = labels == k
            if mask.sum() > 0:
                centroids[k] = embeds[mask].mean(0)
    return centroids, labels


@torch.no_grad()
def evaluate(
    model:   ShapeStratifiedDQE,
    dataset: AirfRANSNearFieldDataset,
    device:  torch.device,
) -> dict[str, float]:
    """MSE and R² on mean-surface-Cp prediction (normalised)."""
    model.eval()
    preds, targets = [], []
    for gg in dataset:
        pred, _, _, _, _ = model(gg.patches.to(device))
        preds.append(pred.item())
        targets.append(get_airfoil_cp(gg).item())
    preds   = np.array(preds)
    targets = np.array(targets)
    mse     = float(np.mean((preds - targets) ** 2))
    ss_res  = float(np.sum((preds - targets) ** 2))
    ss_tot  = float(np.sum((targets - targets.mean()) ** 2))
    r2      = float(1.0 - ss_res / max(ss_tot, 1e-12))
    return {"mse": mse, "r2": r2}


# ── Chord-wise Cp profiles ────────────────────────────────────────────────────


def compute_cp_profile(surf_pos: np.ndarray, surf_cp: np.ndarray) -> np.ndarray:
    """Mean upper+lower chord-wise Cp at N_CHORD stations.

    Mirrors the split logic of ``compute_shape_vector`` but interpolates
    pressure instead of geometry, then averages upper and lower surfaces
    to give the mean chord loading at each station.

    Returns (N_CHORD,) array; zeros on degenerate input.
    """
    if surf_pos.shape[0] < 10:
        return np.zeros(N_CHORD)
    x, y  = surf_pos[:, 0], surf_pos[:, 1]
    chord = x.max() - x.min()
    if chord < 1e-6:
        return np.zeros(N_CHORD)
    x_n   = (x - x.min()) / chord

    y_mid  = y.mean()
    upper  = y >= y_mid
    lower  = y <  y_mid
    if upper.sum() < 6 or lower.sum() < 6:
        return np.zeros(N_CHORD)

    x_grid = np.linspace(0.0, 1.0, N_CHORD)
    su     = np.argsort(x_n[upper])
    cp_up  = np.interp(x_grid, x_n[upper][su], surf_cp[upper][su])
    sl     = np.argsort(x_n[lower])
    cp_lo  = np.interp(x_grid, x_n[lower][sl], surf_cp[lower][sl])
    return (cp_up + cp_lo) / 2.0


def collect_cp_profiles(
    raw_ds:     AirfRANS,
    dataset:    AirfRANSNearFieldDataset,
    p_mean:     float,
    p_std:      float,
) -> np.ndarray:
    """(A, N_CHORD) normalised mean chord-wise Cp for all airfoils in dataset."""
    profiles: list[np.ndarray] = []
    for i in tqdm(range(len(dataset)), desc="  Cp profiles", leave=False, unit="g"):
        raw_idx = dataset._indices[dataset._order[i]]
        raw: Data = raw_ds[raw_idx]    # type: ignore[assignment]
        pos  = cast(torch.Tensor, raw.pos).float().numpy()
        surf = (
            cast(torch.Tensor, raw.surf).bool().numpy()
            if hasattr(raw, "surf") and raw.surf is not None
            else np.zeros(pos.shape[0], dtype=bool)
        )
        p_raw = cast(torch.Tensor, raw.y).float().numpy()[:, P_IDX]
        p_norm = (p_raw - p_mean) / p_std
        profiles.append(compute_cp_profile(pos[surf], p_norm[surf]))
    return np.stack(profiles)


# ── Visualisations ────────────────────────────────────────────────────────────


def plot_training_curves(
    history:    dict[str, list[float]],
    kappa_hist: list[list[float]],
    out_dir:    Path,
) -> None:
    """Loss components and curvature evolution — two separate figures."""
    keys = ["total", "regression", "entropy", "diversity", "centripetal"]

    # ── loss curves ───────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, len(keys), figsize=(5 * len(keys), 4))
    for ax, key in zip(axes, keys):
        if key in history and history[key]:
            ax.plot(history[key])
        ax.set_title(key)
        ax.set_xlabel("epoch (phase 4)")
        ax.set_ylabel("loss")
        ax.grid(lw=0.4, alpha=0.5)
    plt.tight_layout()
    p = out_dir / "training_curves.png"
    plt.savefig(p, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {p}")

    # ── curvature evolution ────────────────────────────────────────────────────
    if not kappa_hist:
        return
    kappa_arr = np.array(kappa_hist)   # (epochs, K)
    fig, ax = plt.subplots(figsize=(8, 4))
    for k in range(kappa_arr.shape[1]):
        ax.plot(kappa_arr[:, k], label=f"Stratum {k}")
    ax.axhline(0, color="k", lw=0.5, ls="--")
    ax.set_xlabel("epoch (phase 4)")
    ax.set_ylabel("κ  (negative=hyperbolic, positive=spherical)")
    ax.set_title("Curvature evolution per stratum")
    ax.legend(loc="best", fontsize=9)
    ax.grid(lw=0.4, alpha=0.5)
    plt.tight_layout()
    p = out_dir / "curvature_evolution.png"
    plt.savefig(p, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {p}")


def plot_stratum_scatter(
    all_embeds: np.ndarray,    # (A, d)
    all_labels: np.ndarray,    # (A,)  hard stratum index
    all_cp:     np.ndarray,    # (A,)  normalised mean surface Cp
    n_strata:   int,
    p_mean:     float,
    p_std:      float,
    out_path:   Path,
) -> None:
    """PCA of airfoil embeddings: left = stratum colour, right = Cp colour."""
    E_c    = all_embeds - all_embeds.mean(0)
    _, _, Vt = np.linalg.svd(E_c, full_matrices=False)
    coords = E_c @ Vt[:2].T   # (A, 2)
    cp_phys = all_cp * p_std + p_mean

    cmap = plt.get_cmap("tab10")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Stratum colour
    ax = axes[0]
    for k in range(n_strata):
        m = all_labels == k
        ax.scatter(
            coords[m, 0], coords[m, 1],
            color=cmap(k / max(n_strata - 1, 1)),
            s=45, alpha=0.85, edgecolors="k", linewidths=0.3,
            label=f"S{k}  (n={m.sum()})",
        )
    ax.set_xlabel("Embedding PC 1")
    ax.set_ylabel("Embedding PC 2")
    ax.set_title("Shape embedding space — coloured by stratum")
    ax.legend(loc="best", fontsize=9)
    ax.grid(lw=0.3, alpha=0.4)

    # Cp colour
    ax = axes[1]
    sc = ax.scatter(
        coords[:, 0], coords[:, 1],
        c=cp_phys, cmap="RdBu_r",
        s=45, edgecolors="k", linewidths=0.3, alpha=0.88,
    )
    fig.colorbar(sc, ax=ax, label="Mean surface Cp")
    ax.set_xlabel("Embedding PC 1")
    ax.set_ylabel("Embedding PC 2")
    ax.set_title("Shape embedding space — coloured by mean Cp")
    ax.grid(lw=0.3, alpha=0.4)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


def plot_stratum_distributions(
    all_labels: np.ndarray,
    all_cp:     np.ndarray,
    n_strata:   int,
    p_mean:     float,
    p_std:      float,
    out_path:   Path,
) -> None:
    """Violin plot of Cp distribution per stratum."""
    cp_phys = all_cp * p_std + p_mean
    data    = [cp_phys[all_labels == k] for k in range(n_strata)]
    filled  = [d if len(d) > 0 else np.array([0.0]) for d in data]

    fig, ax = plt.subplots(figsize=(max(6, 2 * n_strata), 5))
    parts = ax.violinplot(filled, positions=list(range(n_strata)), showmedians=True)
    bodies = cast(list, parts["bodies"])
    for k, pc in enumerate(bodies):
        pc.set_facecolor(plt.get_cmap("tab10")(k / max(n_strata - 1, 1)))
        pc.set_alpha(0.6)
    ax.set_xticks(range(n_strata))
    ax.set_xticklabels([f"S{k}\n(n={len(data[k])})" for k in range(n_strata)])
    ax.set_ylabel("Mean surface Cp")
    ax.set_title("Cp distribution per shape stratum")
    ax.grid(axis="y", lw=0.5, alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


def plot_stratum_cp_profiles(
    all_profiles: np.ndarray,   # (A, N_CHORD)  normalised chord-wise Cp
    all_labels:   np.ndarray,   # (A,)
    n_strata:     int,
    p_mean:       float,
    p_std:        float,
    out_path:     Path,
) -> None:
    """Mean chord-wise Cp profile per stratum with ±1σ shaded band."""
    x_chord = np.linspace(0.0, 1.0, all_profiles.shape[1])
    cmap    = plt.get_cmap("tab10")

    fig, ax = plt.subplots(figsize=(10, 5))
    for k in range(n_strata):
        m = all_labels == k
        if m.sum() == 0:
            continue
        mean_p = all_profiles[m].mean(0) * p_std + p_mean
        std_p  = all_profiles[m].std(0)  * p_std
        c = cmap(k / max(n_strata - 1, 1))
        ax.plot(x_chord, mean_p, color=c, lw=2, label=f"Stratum {k}  (n={m.sum()})")
        ax.fill_between(x_chord, mean_p - std_p, mean_p + std_p, color=c, alpha=0.15)

    ax.axhline(0.0, color="k", lw=0.5, ls="--")
    ax.set_xlabel("Chord position  x/c")
    ax.set_ylabel("Mean surface Cp")
    ax.set_title("Chord-wise Cp profile per stratum  (shaded: ±1 std)")
    ax.legend(loc="best", fontsize=9)
    ax.grid(lw=0.4, alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


def plot_shape_stratum(
    all_shape:  np.ndarray,   # (A, 2*N_CHORD)
    all_labels: np.ndarray,   # (A,)
    all_cp:     np.ndarray,   # (A,)  normalised
    n_strata:   int,
    p_mean:     float,
    p_std:      float,
    out_path:   Path,
) -> None:
    """Geometric shape PCA coloured by stratum; inset r between shape PC1 and Cp.

    Shows whether the strata discovered by the pressure lens correspond to
    geometrically distinct shape families.
    """
    S_c  = all_shape - all_shape.mean(0)
    _, _, Vt = np.linalg.svd(S_c, full_matrices=False)
    coords   = S_c @ Vt[:2].T   # (A, 2)
    cp_phys  = all_cp * p_std + p_mean

    r_val, _ = pearsonr(coords[:, 0], cp_phys)

    cmap = plt.get_cmap("tab10")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Stratum colour
    ax = axes[0]
    for k in range(n_strata):
        m = all_labels == k
        ax.scatter(
            coords[m, 0], coords[m, 1],
            color=cmap(k / max(n_strata - 1, 1)),
            s=45, alpha=0.85, edgecolors="k", linewidths=0.3,
            label=f"S{k}",
        )
    ax.set_xlabel("Geometric shape PC 1  (thickness / camber)")
    ax.set_ylabel("Geometric shape PC 2")
    ax.set_title("Geometric shape space — coloured by pressure stratum")
    ax.legend(loc="best", fontsize=9)
    ax.grid(lw=0.3, alpha=0.4)
    ax.annotate(
        f"Shape PC1 vs Cp\nr = {r_val:+.3f}",
        xy=(0.04, 0.96), xycoords="axes fraction",
        va="top", ha="left", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#999", alpha=0.85),
    )

    # Cp colour
    ax = axes[1]
    sc = ax.scatter(
        coords[:, 0], coords[:, 1],
        c=cp_phys, cmap="RdBu_r",
        s=45, edgecolors="k", linewidths=0.3, alpha=0.88,
    )
    fig.colorbar(sc, ax=ax, label="Mean surface Cp")
    ax.set_xlabel("Geometric shape PC 1  (thickness / camber)")
    ax.set_ylabel("Geometric shape PC 2")
    ax.set_title("Geometric shape space — coloured by mean Cp")
    ax.grid(lw=0.3, alpha=0.4)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


# ── Flow-condition filter ─────────────────────────────────────────────────────


def _u_inf(graph: Data) -> float:
    """Scalar inlet speed |U_inf| (m/s) from the first node's x features."""
    ux = float(cast(torch.Tensor, graph.x)[0, 0])
    uy = float(cast(torch.Tensor, graph.x)[0, 1])
    return math.hypot(ux, uy)


def filter_dataset_indices(
    raw_ds:      AirfRANS,
    reynolds:    float | None,
    mach:        float | None,
    reynolds_tol: float,
    mach_tol:    float,
) -> list[int]:
    """Return indices whose Re and/or Ma are within tolerance of the targets.

    Parameters
    ----------
    reynolds     : target Reynolds number (None = no Re filter)
    mach         : target Mach number    (None = no Ma filter)
    reynolds_tol : fractional tolerance, e.g. 0.05 → ±5 %
    mach_tol     : fractional tolerance, e.g. 0.05 → ±5 %

    Re and Ma are derived from the stored inlet velocity:
      Re = ρ · |U_inf| · chord / μ
      Ma = |U_inf| / c_sound
    using standard sea-level air properties (_RHO, _MU, _C_SND, _CHORD).
    """
    kept: list[int] = []
    for i in range(len(raw_ds)):
        graph: Data = raw_ds[i]   # type: ignore[assignment]
        U   = _u_inf(graph)
        Re  = _RHO * U * _CHORD / _MU
        Ma  = U / _C_SND

        if reynolds is not None and abs(Re - reynolds) > reynolds_tol * reynolds:
            continue
        if mach is not None and abs(Ma - mach) > mach_tol * mach:
            continue
        kept.append(i)
    return kept


# ── Main training loop ────────────────────────────────────────────────────────


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

    # Update run.py's NEAR_FIELD_BOUNDS before any dataset construction
    NEAR_FIELD_BOUNDS["y_min"] = -args.near_field_y
    NEAR_FIELD_BOUNDS["y_max"] =  args.near_field_y

    # ── Data ──────────────────────────────────────────────────────────────────
    print(f"Loading AirfRANS (task={args.task}) …")
    raw_train = AirfRANS(root=args.data_root, task=args.task, train=True)
    raw_test  = AirfRANS(root=args.data_root, task=args.task, train=False)

    # Optional flow-condition filter
    re_filter = args.reynolds if args.reynolds > 0 else None
    ma_filter = args.mach     if args.mach     > 0 else None
    if re_filter is not None or ma_filter is not None:
        label_parts: list[str] = []
        if re_filter is not None:
            label_parts.append(f"Re={re_filter:.3e} ±{args.reynolds_tol*100:.0f}%")
        if ma_filter is not None:
            label_parts.append(f"Ma={ma_filter:.4f} ±{args.mach_tol*100:.0f}%")
        print(f"  Filtering by: {', '.join(label_parts)}")

        train_pool = filter_dataset_indices(
            raw_train, re_filter, ma_filter, args.reynolds_tol, args.mach_tol
        )
        test_pool  = filter_dataset_indices(
            raw_test,  re_filter, ma_filter, args.reynolds_tol, args.mach_tol
        )
        if len(train_pool) == 0:
            raise ValueError(
                "Flow-condition filter matched 0 training samples. "
                "Widen --reynolds_tol / --mach_tol or adjust the target values."
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
    print(f"  Pressure  mean={p_mean_v:.4f}  std={p_std_v:.4f}")

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
        norm_stats=norm_stats, indices=test_pool, _shared_raw=raw_test,
    )
    print(
        f"  {len(train_ds)} train / {len(val_ds)} val / {len(test_ds)} test\n"
        f"  Grid {GRID_H}×{GRID_W}  Patches/airfoil: {N_PATCHES}"
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = ShapeStratifiedDQE(
        embed_dim     = args.embed_dim,
        n_strata      = args.n_strata,
        n_intervals   = args.n_intervals,
        n_protos      = args.n_protos,
        base_channels = args.base_channels,
        dropout       = args.dropout,
        K_max         = args.K_max,
        interval_temp = args.interval_temp,
        proto_temp    = args.proto_temp,
    ).to(device)

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        print(f"Resumed from {args.resume}")

    history: dict[str, list[float]] = {
        k: [] for k in ["total", "regression", "entropy", "diversity", "centripetal"]
    }
    kappa_hist: list[list[float]] = []

    # ── Phase 1: Warmup ────────────────────────────────────────────────────────
    print(f"\n=== Phase 1: Warmup ({args.warmup_epochs} epochs) ===")
    model.set_phase(1)
    opt1 = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=1e-4,
    )
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt1.step()
            ep_loss += loss.item()
            del patches_d, target_d, pred
        sched1.step()
        if (epoch + 1) % max(1, args.warmup_epochs // 5) == 0:
            print(f"  [warmup] epoch {epoch + 1:3d}  loss={ep_loss / len(train_ds):.4f}")

    # ── Phase 2: K-means ───────────────────────────────────────────────────────
    print("\n=== Phase 2: K-means clustering ===")
    torch.cuda.empty_cache()
    all_embeds_kmeans = collect_pooled_embeddings(model, train_ds, device)
    centres, labels   = kmeans_cluster(all_embeds_kmeans, args.n_strata, seed=args.seed)
    model.init_from_kmeans(centres, labels, all_embeds_kmeans)
    sizes = [(labels == k).sum().item() for k in range(args.n_strata)]
    print(f"  Cluster sizes: {sizes}")

    # ── Phase 3: Assigner pre-train ────────────────────────────────────────────
    print(f"\n=== Phase 3: Assigner pre-train ({args.assigner_epochs} epochs) ===")
    model.set_phase(2)
    opt2 = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr * 2, weight_decay=1e-4,
    )
    centres_d = centres.to(device)

    for epoch in tqdm(range(args.assigner_epochs), desc="assigner", unit="ep"):
        model.train()
        ep_loss = 0.0
        train_ds.shuffle()
        for gg in train_ds:
            patches_d = gg.patches.to(device)
            opt2.zero_grad()
            with torch.no_grad():
                g = model.pool_patches(patches_d)
                diff = g - centres_d.unsqueeze(0)    # (1, K, d)
                pseudo = (diff ** 2).sum(-1).argmin(-1)  # (1,)
            soft = model.assigner(g.detach())
            loss = F.nll_loss(soft.clamp(1e-8).log(), pseudo)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt2.step()
            ep_loss += loss.item()
            del patches_d, g, soft
        if (epoch + 1) % max(1, args.assigner_epochs // 5) == 0:
            print(f"  [assigner] epoch {epoch + 1:3d}  ce={ep_loss / len(train_ds):.4f}")

    # ── Phase 4: Full joint training ───────────────────────────────────────────
    print(f"\n=== Phase 4: Full joint training ({args.full_epochs} epochs) ===")
    model.set_phase(3)
    opt3 = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=1e-4,
    )
    sched3 = CosineAnnealingLR(opt3, T_max=max(1, args.full_epochs))

    best_val_mse = math.inf
    best_epoch   = 0

    epoch_bar = tqdm(range(args.full_epochs), desc="full DQE", unit="ep")
    for epoch in epoch_bar:
        model.train()
        ep: dict[str, float] = {k: 0.0 for k in history}
        train_ds.shuffle()

        for gg in train_ds:
            patches_d = gg.patches.to(device)
            target_d  = get_airfoil_cp(gg).to(device)

            opt3.zero_grad()
            pred, soft, g, protos_tan, kappas_t = model(patches_d)

            l_reg  = F.mse_loss(pred, target_d)
            l_ent  = stratum_entropy_loss(soft)
            l_div  = curvature_diversity_loss(kappas_t, args.curvature_margin)
            # Centripetal: pull airfoil embedding toward its expected prototype
            protos_mean = protos_tan.mean(dim=1)          # (K, d)
            l_cent = centripetal_loss(g, protos_mean, soft)

            loss = (
                l_reg
                + args.entropy_weight     * l_ent
                + args.diversity_weight   * l_div
                + args.centripetal_weight * l_cent
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt3.step()

            ep["regression"]  += l_reg.item()
            ep["entropy"]     += l_ent.item()
            ep["diversity"]   += l_div.item()
            ep["centripetal"] += l_cent.item()
            ep["total"]       += loss.item()
            del patches_d, target_d, pred, soft, g, protos_tan

        sched3.step()
        ng = max(1, len(train_ds))
        for k in history:
            history[k].append(ep[k] / ng)

        kappas_now = [
            dqe.kappa.item() for dqe in model.dqes if isinstance(dqe, StratumDQE)
        ]
        kappa_hist.append(kappas_now)

        epoch_bar.set_postfix(
            reg=f"{history['regression'][-1]:.4f}",
            ent=f"{history['entropy'][-1]:.4f}",
        )

        if (epoch + 1) % max(1, args.full_epochs // 10) == 0:
            torch.cuda.empty_cache()
            val_m = evaluate(model, val_ds, device)
            epoch_bar.write(
                f"  epoch {epoch + 1:3d}  "
                f"val_mse={val_m['mse']:.4f}  val_r2={val_m['r2']:.3f}  "
                f"κ={[f'{k:.2f}' for k in kappas_now]}"
            )
            if val_m["mse"] < best_val_mse:
                best_val_mse = val_m["mse"]
                best_epoch   = epoch + 1
                torch.save(
                    {
                        "model": model.state_dict(),
                        "epoch": epoch + 1,
                        "val_metrics": val_m,
                        "norm_stats": {
                            "x_mean": x_mean.cpu(), "x_std": x_std.cpu(),
                            "p_mean": p_mean_t.cpu(), "p_std": p_std_t.cpu(),
                        },
                    },
                    out_dir / "checkpoint.pt",
                )

    print(f"\nBest val MSE = {best_val_mse:.4f} at epoch {best_epoch}")

    # ── Final evaluation ───────────────────────────────────────────────────────
    print("\n=== Final evaluation on test set ===")
    ckpt = torch.load(out_dir / "checkpoint.pt", map_location=device)
    model.load_state_dict(ckpt["model"])
    torch.cuda.empty_cache()

    test_m = evaluate(model, test_ds, device)
    for k, v in test_m.items():
        print(f"  {k}: {v:.4f}")
    with open(out_dir / "test_metrics.json", "w") as f:
        json.dump(test_m, f, indent=2)

    # ── Collect representations ────────────────────────────────────────────────
    print("\n=== Collecting representations for visualisation ===")
    torch.cuda.empty_cache()
    model.eval()

    embs_tr, softs_tr, cps_tr = collect_airfoil_embeddings(model, train_ds, device)
    embs_va, softs_va, cps_va = collect_airfoil_embeddings(model, val_ds,   device)
    embs_te, softs_te, cps_te = collect_airfoil_embeddings(model, test_ds,  device)

    all_embs   = torch.cat([embs_tr, embs_va, embs_te]).numpy().astype(np.float64)
    all_softs  = torch.cat([softs_tr, softs_va, softs_te]).numpy()
    all_cps    = torch.cat([cps_tr, cps_va, cps_te]).numpy()
    all_labels = all_softs.argmax(axis=1)   # hard stratum assignments
    print(f"  {len(all_labels)} airfoils  embed_dim={all_embs.shape[1]}")

    # Geometric shape vectors
    print("  Computing geometric shape vectors …")
    shape_tr = collect_shape_vectors(raw_train, train_ds)
    shape_va = collect_shape_vectors(raw_train, val_ds)
    shape_te = collect_shape_vectors(raw_test,  test_ds)
    all_shape = np.concatenate([shape_tr, shape_va, shape_te])

    # Chord-wise Cp profiles
    print("  Computing chord-wise Cp profiles …")
    prof_tr = collect_cp_profiles(raw_train, train_ds, p_mean_v, p_std_v)
    prof_va = collect_cp_profiles(raw_train, val_ds,   p_mean_v, p_std_v)
    prof_te = collect_cp_profiles(raw_test,  test_ds,  p_mean_v, p_std_v)
    all_profiles = np.concatenate([prof_tr, prof_va, prof_te])

    # ── Visualisations ─────────────────────────────────────────────────────────
    print("\n=== Saving visualisations ===")

    plot_training_curves(history, kappa_hist, out_dir)

    plot_stratum_scatter(
        all_embs, all_labels, all_cps,
        args.n_strata, p_mean_v, p_std_v,
        out_dir / "stratum_scatter.png",
    )
    plot_stratum_distributions(
        all_labels, all_cps,
        args.n_strata, p_mean_v, p_std_v,
        out_dir / "stratum_distributions.png",
    )
    plot_stratum_cp_profiles(
        all_profiles, all_labels,
        args.n_strata, p_mean_v, p_std_v,
        out_dir / "stratum_cp_profiles.png",
    )
    plot_shape_stratum(
        all_shape, all_labels, all_cps,
        args.n_strata, p_mean_v, p_std_v,
        out_dir / "shape_stratum.png",
    )

    print(f"\nAll outputs saved to {out_dir}")


# ── CLI ───────────────────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Shape-Stratified DQE for AirfRANS — pressure-supervised shape clustering",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Data
    p.add_argument("--data_root",    default="../data/AirfRANS")
    p.add_argument("--task",         default="scarce", choices=["scarce", "full"])
    p.add_argument("--near_field_y", type=float, default=0.6,
                   help="Half-height of near-field domain; must match run.py")
    # Model
    p.add_argument("--n_strata",      type=int,   default=4)
    p.add_argument("--embed_dim",     type=int,   default=64)
    p.add_argument("--n_intervals",   type=int,   default=8)
    p.add_argument("--n_protos",      type=int,   default=8)
    p.add_argument("--base_channels", type=int,   default=32,
                   help="GeometricConvAutoencoder base filter count")
    p.add_argument("--dropout",       type=float, default=0.1)
    p.add_argument("--K_max",         type=float, default=2.0)
    p.add_argument("--interval_temp", type=float, default=0.5)
    p.add_argument("--proto_temp",    type=float, default=1.0)
    # Training
    p.add_argument("--warmup_epochs",   type=int,   default=30)
    p.add_argument("--assigner_epochs", type=int,   default=10)
    p.add_argument("--full_epochs",     type=int,   default=100)
    p.add_argument("--lr",              type=float, default=3e-4)
    # Loss
    p.add_argument("--entropy_weight",     type=float, default=0.1)
    p.add_argument("--diversity_weight",   type=float, default=0.05)
    p.add_argument("--centripetal_weight", type=float, default=0.01)
    p.add_argument("--curvature_margin",   type=float, default=0.1)
    # Flow-condition filter (0 = disabled)
    p.add_argument(
        "--reynolds", type=float, default=0,
        help="Target Reynolds number; 0 = no filter  (e.g. 3e6)",
    )
    p.add_argument(
        "--reynolds_tol", type=float, default=0.05,
        help="Fractional tolerance for Re filter  (default ±5%%)",
    )
    p.add_argument(
        "--mach", type=float, default=0,
        help="Target Mach number; 0 = no filter  (e.g. 0.15)",
    )
    p.add_argument(
        "--mach_tol", type=float, default=0.05,
        help="Fractional tolerance for Ma filter  (default ±5%%)",
    )
    # Misc
    p.add_argument("--out_dir", default="results/shape_strata")
    p.add_argument("--seed",    type=int, default=42)
    p.add_argument("--cpu",     action="store_true")
    p.add_argument("--resume",  default="", help="Checkpoint to resume from")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    t0 = time.time()
    train(args)
    print(f"\nTotal wall time: {(time.time() - t0) / 60:.1f} min")
