"""
Visualisation utilities for StratifiedDQE training on AirfRANS.

Functions
---------
plot_training_curves     — multi-panel loss curves (total + components)
plot_curvature_evolution — κ_k vs epoch for all K strata
plot_stratum_assignments — 2-D UMAP of node embeddings coloured by stratum
plot_prediction_scatter  — scatter pred vs truth for each target variable
plot_geometry_summary    — bar chart of geometry type distribution per stratum
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm


# ── colour palette ──────────────────────────────────────────────────────────

STRATUM_COLORS = [
    "#e41a1c", "#377eb8", "#4daf4a", "#984ea3",
    "#ff7f00", "#a65628", "#f781bf", "#999999",
]

TARGET_NAMES = ["u_x", "u_y", "p", "ν_t"]


# ── Training curves ──────────────────────────────────────────────────────────

def plot_training_curves(
    history:   dict[str, list[float]],  # {"total": [...], "regression": [...], ...}
    save_path: str | Path,
    phase_boundaries: Optional[list[int]] = None,
    title: str = "StratifiedDQE Training",
) -> None:
    """Plot loss components over epochs on a shared x-axis."""
    keys = list(history.keys())
    n    = len(keys)
    fig, axes = plt.subplots(n, 1, figsize=(10, 2.5 * n), sharex=True)
    if n == 1:
        axes = [axes]

    epochs = np.arange(1, len(history[keys[0]]) + 1)
    for ax, key in zip(axes, keys):
        vals = np.array(history[key], dtype=float)
        ax.plot(epochs, vals, lw=1.5)
        ax.set_ylabel(key)
        ax.set_yscale("log" if vals.min() > 0 else "linear")
        if phase_boundaries:
            for pb in phase_boundaries:
                ax.axvline(pb, color="gray", ls="--", lw=0.8, alpha=0.7)
        ax.grid(True, which="both", alpha=0.3)

    axes[-1].set_xlabel("Epoch")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Curvature evolution ──────────────────────────────────────────────────────

def plot_curvature_evolution(
    kappa_history: list[list[float]],  # list over epochs of list over strata
    save_path: str | Path,
    kappa_eps: float = 5e-3,
) -> None:
    """Plot κ_k vs epoch for each stratum.  Shade Euclidean band |κ|<kappa_eps."""
    K      = len(kappa_history[0])
    epochs = np.arange(1, len(kappa_history) + 1)
    arr    = np.array(kappa_history)  # (E, K)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.axhspan(-kappa_eps, kappa_eps, color="lightgrey", alpha=0.5, label="Euclidean band")
    ax.axhline(0, color="k", lw=0.5)

    for k in range(K):
        color = STRATUM_COLORS[k % len(STRATUM_COLORS)]
        ax.plot(epochs, arr[:, k], lw=1.5, color=color, label=f"stratum {k}")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("κ")
    ax.set_title("Curvature evolution per stratum")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Stratum assignments on UMAP ─────────────────────────────────────────────

def plot_stratum_assignments(
    embeds:      torch.Tensor,   # (N, d) — node embeddings
    assignments: torch.Tensor,   # (N,)   — hard stratum labels
    save_path:   str | Path,
    surf_mask:   Optional[torch.Tensor] = None,  # (N,) bool — surface nodes
    title:       str = "Stratum assignments (UMAP)",
    max_points:  int = 20_000,
) -> None:
    """Visualise node embeddings in 2-D via UMAP, coloured by stratum."""
    try:
        from umap import UMAP
    except ImportError:
        _plot_stratum_pca(embeds, assignments, save_path, surf_mask, title, max_points)
        return

    emb_np = embeds.detach().cpu().numpy().astype(np.float32)
    asgn   = assignments.detach().cpu().numpy()

    # Subsample for speed
    if emb_np.shape[0] > max_points:
        idx    = np.random.choice(emb_np.shape[0], max_points, replace=False)
        emb_np = emb_np[idx]
        asgn   = asgn[idx]
        surf_mask_np = surf_mask.cpu().numpy()[idx] if surf_mask is not None else None
    else:
        surf_mask_np = surf_mask.cpu().numpy() if surf_mask is not None else None

    reducer = UMAP(n_components=2, random_state=42, min_dist=0.1, n_neighbors=30)
    umap2   = reducer.fit_transform(emb_np)

    K = int(asgn.max()) + 1
    fig, ax = plt.subplots(figsize=(8, 6))
    for k in range(K):
        mask = asgn == k
        ax.scatter(
            umap2[mask, 0], umap2[mask, 1],
            s=2, alpha=0.4,
            color=STRATUM_COLORS[k % len(STRATUM_COLORS)],
            label=f"stratum {k}",
        )
    if surf_mask_np is not None:
        ax.scatter(
            umap2[surf_mask_np, 0], umap2[surf_mask_np, 1],
            s=6, alpha=0.9, color="black", marker="x", label="surface",
        )

    ax.set_title(title)
    ax.legend(loc="best", fontsize=8, markerscale=4)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_stratum_pca(embeds, assignments, save_path, surf_mask, title, max_points):
    """Fallback: PCA instead of UMAP."""
    from sklearn.decomposition import PCA

    emb_np = embeds.detach().cpu().numpy().astype(np.float32)
    asgn   = assignments.detach().cpu().numpy()

    if emb_np.shape[0] > max_points:
        idx    = np.random.choice(emb_np.shape[0], max_points, replace=False)
        emb_np = emb_np[idx]
        asgn   = asgn[idx]
        surf_mask = surf_mask[idx] if surf_mask is not None else None

    pca2 = PCA(n_components=2).fit_transform(emb_np)

    K = int(asgn.max()) + 1
    fig, ax = plt.subplots(figsize=(8, 6))
    for k in range(K):
        m = asgn == k
        ax.scatter(pca2[m, 0], pca2[m, 1], s=2, alpha=0.4,
                   color=STRATUM_COLORS[k % len(STRATUM_COLORS)],
                   label=f"stratum {k}")
    ax.set_title(title + " (PCA fallback)")
    ax.legend(fontsize=8, markerscale=4)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Prediction scatter ───────────────────────────────────────────────────────

def plot_prediction_scatter(
    pred:      torch.Tensor,   # (N, T)
    target:    torch.Tensor,   # (N, T)
    save_path: str | Path,
    surf_mask: Optional[torch.Tensor] = None,
    target_names: list[str] = TARGET_NAMES,
    max_points: int = 10_000,
) -> None:
    """Scatter plot of predicted vs true values for each target."""
    pred_np   = pred.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()
    T         = pred_np.shape[1]

    if pred_np.shape[0] > max_points:
        idx      = np.random.choice(pred_np.shape[0], max_points, replace=False)
        pred_np   = pred_np[idx]
        target_np = target_np[idx]
        surf_np   = surf_mask.cpu().numpy()[idx] if surf_mask is not None else None
    else:
        surf_np = surf_mask.cpu().numpy() if surf_mask is not None else None

    ncols = min(T, 4)
    nrows = math.ceil(T / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    axes = np.array(axes).flatten() if T > 1 else [axes]

    names = (target_names + [f"target_{i}" for i in range(T)])[:T]
    for i, (ax, name) in enumerate(zip(axes, names)):
        p = pred_np[:, i]
        t = target_np[:, i]
        vmin, vmax = min(p.min(), t.min()), max(p.max(), t.max())
        ax.scatter(t, p, s=1, alpha=0.3, c="steelblue", rasterized=True)
        if surf_np is not None:
            ax.scatter(t[surf_np], p[surf_np], s=3, alpha=0.7, c="red",
                       rasterized=True, label="surface")
        ax.plot([vmin, vmax], [vmin, vmax], "k--", lw=0.8)
        r2 = float(1.0 - np.var(p - t) / (np.var(t) + 1e-12))
        ax.set_title(f"{name}  R²={r2:.3f}")
        ax.set_xlabel("true")
        ax.set_ylabel("pred")
        if surf_np is not None:
            ax.legend(fontsize=7, markerscale=3)

    for ax in axes[T:]:
        ax.set_visible(False)

    fig.suptitle("Prediction vs Truth", fontsize=12)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Geometry summary ─────────────────────────────────────────────────────────

def plot_geometry_summary(
    kappas:    list[float],   # one per stratum (at end of training)
    save_path: str | Path,
    kappa_eps: float = 5e-3,
) -> None:
    """Bar chart of final curvature per stratum, coloured by geometry type."""
    K = len(kappas)
    x = np.arange(K)
    colors = []
    for k in kappas:
        if k < -kappa_eps:
            colors.append("#4292c6")    # blue = hyperbolic
        elif k > kappa_eps:
            colors.append("#e6550d")    # orange = spherical
        else:
            colors.append("#74c476")    # green = flat

    fig, ax = plt.subplots(figsize=(max(5, K * 0.8), 4))
    bars = ax.bar(x, kappas, color=colors, edgecolor="k", linewidth=0.5)
    ax.axhspan(-kappa_eps, kappa_eps, color="lightgrey", alpha=0.5, label="Euclidean band")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"Stratum {i}" for i in range(K)], rotation=30, ha="right")
    ax.set_ylabel("κ")
    ax.set_title("Final curvature per stratum")

    from matplotlib.patches import Patch
    legend_elems = [
        Patch(facecolor="#4292c6", label="Hyperbolic (κ < 0)"),
        Patch(facecolor="#74c476", label="Flat (|κ| ≤ ε)"),
        Patch(facecolor="#e6550d", label="Spherical (κ > 0)"),
    ]
    ax.legend(handles=legend_elems, fontsize=8)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── import math at module level ──────────────────────────────────────────────
import math
