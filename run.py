"""
run.py — Pressure-DQE on near-airfoil patches → 3-D pressure-similarity surface.

Differences from run_airfrans_cnn.py
--------------------------------------
Near-field filtering
    The interpolation domain is clipped to a tight box around the airfoil
    (NEAR_FIELD_BOUNDS, default x∈[-0.25,1.5] y∈[-0.4,0.4]).  This gives
    roughly 4× better spatial resolution per grid cell compared with the
    full-domain bounds, so patches resolve boundary-layer and near-wake
    physics rather than far-field quiescent flow.

Pressure-only lens
    Only pressure (channel 2 of the AirfRANS 4-target output) is predicted.
    The single DQE head learns an interval/prototype decomposition of the
    pressure field; embeddings therefore encode pressure-distribution shape.

No stratification
    StratumAssigner and k-means phase are removed.  One StratumDQE head
    trains directly on pressure with two phases:
      1. Warmup  — encoder + linear regression head (MSE on pressure)
      2. Full    — encoder + StratumDQE (pressure MSE + centripetal)

3-D surface visualisation
    After training, each airfoil's patch embeddings are averaged to give one
    vector per airfoil.  PCA projects these to 2-D for the x-y plane; mean
    surface pressure provides the z-axis.  A surface interpolated over the
    scatter forms a "pressure landscape" where proximity reflects aerodynamic
    similarity and height reflects mean pressure level.

Usage
-----
  python run.py --out_dir results/pressure_dqe

  # Quick smoke test
  python run.py --n_intervals 4 --n_protos 4 --warmup_epochs 5 --full_epochs 10
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.interpolate import griddata
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.datasets import AirfRANS
from torch_geometric.data import Data

from models.GeometricCNNAutoencoder import GeometricPatchExtractor, ReferenceDomain
from StratifiedEmbedding import StratumDQE, WarmupRegressor

# ── Constants ─────────────────────────────────────────────────────────────────

# Tight near-field box: captures the airfoil body (chord 0→1 at y≈0), a
# quarter-chord upstream buffer, half-chord downstream near-wake, and
# enough vertical extent to cover the wake at high angles of attack.
# y_min/y_max are overridden at runtime by --near_field_y (default ±0.6).
# AirfRANS "scarce" goes up to ~20° AoA; at 20° the wake can reach y≈±0.6.
# LinearNDInterpolator fills grid cells beyond the CFD convex hull with 0,
# so too-tight bounds silently corrupt patches near the domain edge.
NEAR_FIELD_BOUNDS: dict[str, float] = {
    "x_min": -0.25,
    "x_max": 1.5,
    "y_min": -0.6,
    "y_max": 0.6,
}

GRID_H: int = 128
GRID_W: int = 128
PATCH_H: int = 32
PATCH_W: int = 32
PATCH_STRIDE: int = 16

N_PATCHES_H: int = (GRID_H - PATCH_H) // PATCH_STRIDE + 1  # 7
N_PATCHES_W: int = (GRID_W - PATCH_W) // PATCH_STRIDE + 1  # 7
N_PATCHES: int = N_PATCHES_H * N_PATCHES_W  # 49

IN_CHANNELS: int = 7   # pos(2) + AirfRANS x features(5)
P_IDX: int = 2         # pressure index in AirfRANS y = (u_x, u_y, **p**, nu_t)


# ── Data container ────────────────────────────────────────────────────────────


@dataclass
class GraphGrid:
    """Near-field patch representation for one AirfRANS simulation.

    Node-level arrays are NOT cached here — they are recomputed on demand by
    AirfRANSNearFieldDataset.get_node_pressure_eval() to keep the per-entry
    cache size small (~14 MB patches vs ~21 MB if node arrays were included).

    Attributes
    ----------
    patches        : (P, C, h, w) — normalised CNN input patches
    patch_pressure : (P,)          — mean normalised pressure per patch
    patch_surf     : (P,)  bool    — True if any surface node falls in the patch
    positions      : (P, 2) int    — (row, col) top-left of each patch in the grid
    """

    patches: torch.Tensor
    patch_pressure: torch.Tensor
    patch_surf: torch.Tensor
    positions: torch.Tensor


# ── Dataset ───────────────────────────────────────────────────────────────────


class AirfRANSNearFieldDataset:
    """Lazy per-sample loader: point cloud → near-field grid → patches.

    On first access to graph *i* the CFD point cloud is interpolated onto a
    GRID_H×GRID_W grid clipped to NEAR_FIELD_BOUNDS and stored in a RAM cache.
    Node-level arrays needed only at evaluation time are recomputed from the
    raw dataset on demand (not cached) via get_node_pressure_eval().

    Share _shared_raw between train/val/test views that use the same split to
    avoid duplicating the in-memory AirfRANS dataset.  Share _shared_cache
    between train and val (same raw + same norm_stats) so each graph is
    interpolated at most once.

    Parameters
    ----------
    norm_stats : (x_mean, x_std, p_mean, p_std) | None
    _shared_raw : pre-loaded AirfRANS instance; a new one is created if None
    _shared_cache : dict[int, GraphGrid] shared between views; fresh dict if None
    """

    def __init__(
        self,
        root: str,
        split: str,
        task: str = "scarce",
        norm_stats: tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
        ] | None = None,
        indices: list[int] | None = None,
        _shared_raw: AirfRANS | None = None,
        _shared_cache: dict[int, "GraphGrid"] | None = None,
    ) -> None:
        self._raw: AirfRANS = (
            _shared_raw
            if _shared_raw is not None
            else AirfRANS(root=root, task=task, train=(split == "train"))
        )
        n = len(self._raw)
        self._indices: list[int] = (
            list(indices) if indices is not None else list(range(n))
        )
        self._order: list[int] = list(range(len(self._indices)))
        self.norm_stats = norm_stats

        self._ref_domain = ReferenceDomain(
            grid_size=(GRID_H, GRID_W), bounds=NEAR_FIELD_BOUNDS
        )
        self._extractor = GeometricPatchExtractor(
            patch_size=(PATCH_H, PATCH_W), stride=PATCH_STRIDE
        )
        self._cache: dict[int, GraphGrid] = (
            _shared_cache if _shared_cache is not None else {}
        )

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, pos: int) -> GraphGrid:
        raw_idx = self._indices[self._order[pos]]
        if raw_idx not in self._cache:
            self._cache[raw_idx] = self._build_grid(raw_idx)
        return self._cache[raw_idx]

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def shuffle(self) -> None:
        random.shuffle(self._order)

    def get_node_pressure_eval(
        self, pos: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (node_pressure_norm, node_surf, node_grid_rc) — not cached.

        Recomputed from raw data each call so the GraphGrid cache stays small.
        """
        raw_idx = self._indices[self._order[pos]]
        raw: Data = self._raw[raw_idx]  # type: ignore[assignment]

        pos_ = cast(torch.Tensor, raw.pos).float()
        p = cast(torch.Tensor, raw.y).float()[:, P_IDX]  # (N,)
        surf = (
            cast(torch.Tensor, raw.surf).bool()
            if hasattr(raw, "surf") and raw.surf is not None
            else torch.zeros(pos_.shape[0], dtype=torch.bool)
        )

        if self.norm_stats is not None:
            _, _, p_mean, p_std = self.norm_stats
            p = (p - p_mean) / p_std

        x_min = NEAR_FIELD_BOUNDS["x_min"]
        x_max = NEAR_FIELD_BOUNDS["x_max"]
        y_min = NEAR_FIELD_BOUNDS["y_min"]
        y_max = NEAR_FIELD_BOUNDS["y_max"]
        col_f = (pos_[:, 0] - x_min) / (x_max - x_min) * (GRID_W - 1)
        row_f = (pos_[:, 1] - y_min) / (y_max - y_min) * (GRID_H - 1)
        col_i = col_f.round().long().clamp(0, GRID_W - 1)
        row_i = row_f.round().long().clamp(0, GRID_H - 1)
        node_grid_rc = torch.stack([row_i, col_i], dim=-1)  # (N, 2)

        return p, surf, node_grid_rc

    # ── internals ─────────────────────────────────────────────────────────────

    def _build_grid(self, raw_idx: int) -> GraphGrid:
        """Interpolate one AirfRANS graph onto the near-field grid."""
        raw: Data = self._raw[raw_idx]  # type: ignore[assignment]

        pos_ = cast(torch.Tensor, raw.pos).float()
        x_feat = cast(torch.Tensor, raw.x).float()
        p_raw = cast(torch.Tensor, raw.y).float()[:, P_IDX : P_IDX + 1]  # (N, 1)
        surf = (
            cast(torch.Tensor, raw.surf).bool()
            if hasattr(raw, "surf") and raw.surf is not None
            else torch.zeros(pos_.shape[0], dtype=torch.bool)
        )

        x_aug = torch.cat([pos_, x_feat], dim=-1)  # (N, 7)

        if self.norm_stats is not None:
            x_mean, x_std, p_mean, p_std = self.norm_stats
            x_aug = (x_aug - x_mean) / x_std
            p_raw = (p_raw - p_mean) / p_std

        pos_np = pos_.numpy()
        x_np = x_aug.numpy()
        p_np = p_raw.numpy()        # (N, 1)
        s_np = surf.float().numpy() # (N,)

        # One LinearNDInterpolator per call — triangulation built once inside
        grid_x = self._ref_domain.interpolate_field(pos_np, x_np)  # (H, W, 7)
        grid_p = self._ref_domain.interpolate_field(pos_np, p_np)  # (H, W, 1)
        grid_s = self._ref_domain.interpolate_field(pos_np, s_np)  # (H, W)

        flow_patches_np, _, positions_list, _ = (
            self._extractor.extract_patches_with_positions(grid_x)
        )  # (P, h, w, C)

        P = len(positions_list)
        positions_t = torch.tensor(positions_list, dtype=torch.long)
        patches_t = torch.from_numpy(flow_patches_np).float().permute(0, 3, 1, 2)

        grid_p_t = torch.from_numpy(grid_p).float().squeeze(-1)  # (H, W)
        grid_s_t = torch.from_numpy(grid_s).float()               # (H, W)

        patch_pressure = torch.zeros(P, dtype=torch.float32)
        patch_surf = torch.zeros(P, dtype=torch.bool)
        for pi, (ri, ci) in enumerate(positions_list):
            patch_pressure[pi] = grid_p_t[ri : ri + PATCH_H, ci : ci + PATCH_W].mean()
            patch_surf[pi] = grid_s_t[ri : ri + PATCH_H, ci : ci + PATCH_W].max() > 0.1

        return GraphGrid(
            patches=patches_t,
            patch_pressure=patch_pressure,
            patch_surf=patch_surf,
            positions=positions_t,
        )


# ── Normalisation ─────────────────────────────────────────────────────────────


def compute_normalisation(
    raw_ds: AirfRANS,
    indices: list[int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Streaming (x_mean, x_std, p_mean, p_std) over the training set.

    Operates directly on raw point-cloud data before grid interpolation so
    normalisation is consistent with what the dataset wrapper applies.
    """
    first: Data = raw_ds[indices[0]]  # type: ignore[assignment]
    pos0 = cast(torch.Tensor, first.pos).float()
    x0 = cast(torch.Tensor, first.x).float()
    Fx = pos0.shape[1] + x0.shape[1]

    x_sum = torch.zeros(Fx)
    x_sum2 = torch.zeros(Fx)
    p_sum = torch.tensor(0.0)
    p_sum2 = torch.tensor(0.0)
    n_x = n_p = 0

    for idx in indices:
        raw: Data = raw_ds[idx]  # type: ignore[assignment]
        pos_ = cast(torch.Tensor, raw.pos).float()
        x_ = cast(torch.Tensor, raw.x).float()
        p_ = cast(torch.Tensor, raw.y).float()[:, P_IDX]
        x_aug = torch.cat([pos_, x_], dim=-1)

        x_sum += x_aug.sum(0)
        x_sum2 += (x_aug ** 2).sum(0)
        p_sum += p_.sum()
        p_sum2 += (p_ ** 2).sum()
        n_x += x_aug.shape[0]
        n_p += p_.shape[0]

    x_mean = x_sum / n_x
    x_std = ((x_sum2 / n_x) - x_mean ** 2).clamp(min=0).sqrt().clamp(min=1e-6)
    p_mean = p_sum / n_p
    p_std = ((p_sum2 / n_p) - p_mean ** 2).clamp(min=0).sqrt().clamp(min=1e-6)

    return x_mean, x_std, p_mean, p_std


# ── Model ─────────────────────────────────────────────────────────────────────


class PatchCNNEncoder(nn.Module):
    """Three stride-2 conv blocks → AdaptiveAvgPool2d → linear projection.

    LayerNorm-free (uses SiLU non-linearities) for consistency with
    run_airfrans_cnn.py.  Works at batch size 1 (single-graph inference).
    """

    def __init__(
        self,
        in_channels: int,
        embed_dim: int,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        d = hidden_dim
        self.convs = nn.Sequential(
            nn.Conv2d(in_channels, d // 4, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(d // 4, d // 2, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(d // 2, d, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(d, embed_dim, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, C, h, w) → (B, d)
        return self.proj(self.dropout(self.convs(x).squeeze(-1).squeeze(-1)))


class PressureDQEModel(nn.Module):
    """CNN encoder + single DQE head predicting pressure.

    Removing stratification simplifies training to two phases:
      Phase 1 — encoder + WarmupRegressor (plain MSE on pressure)
      Phase 2 — full model: encoder + StratumDQE
                loss = surf-weighted pressure MSE + centripetal

    The StratumDQE's interval/prototype decomposition learns to partition the
    pressure distribution into meaningful quantile regions.  Post-training,
    ``encode()`` maps any set of airfoil patches to embeddings whose geometry
    reflects pressure-distribution similarity.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        embed_dim: int,
        n_intervals: int,
        n_protos: int,
        dropout: float = 0.1,
        K_max: float = 2.0,
        interval_temp: float = 0.5,
        proto_temp: float = 1.0,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.encoder = PatchCNNEncoder(in_channels, embed_dim, hidden_dim, dropout)
        self.dqe = StratumDQE(
            embed_dim, n_intervals, n_protos,
            n_targets=1,
            K_max=K_max,
            interval_temp=interval_temp,
            proto_temp=proto_temp,
        )
        self.warmup_head = WarmupRegressor(embed_dim, n_targets=1)

    def set_phase(self, phase: int) -> None:
        for p in self.parameters():
            p.requires_grad_(False)
        if phase == 1:
            for p in self.encoder.parameters():
                p.requires_grad_(True)
            for p in self.warmup_head.parameters():
                p.requires_grad_(True)
        elif phase == 2:
            for p in self.parameters():
                p.requires_grad_(True)

    def encode(self, patches: torch.Tensor) -> torch.Tensor:
        return self.encoder(patches)

    def warmup_forward(self, patches: torch.Tensor) -> torch.Tensor:
        return self.warmup_head(self.encode(patches))

    def forward(
        self, patches: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (pred_pressure (P,1), embeds (P,d), iv_w (P,I), pr_w (P,P_proto))."""
        emb = self.encode(patches)
        pred, iv_w, pr_w = self.dqe(emb)
        return pred, emb, iv_w, pr_w


# ── Evaluation ────────────────────────────────────────────────────────────────


def _reconstruct_node_predictions(
    patch_preds: torch.Tensor,  # (P,) scalar per patch
    positions: torch.Tensor,    # (P, 2) int — (row, col) top-left
    node_grid_rc: torch.Tensor, # (N, 2) int — (row, col) per node
) -> torch.Tensor:              # (N,)
    """Average overlapping patch predictions onto the grid, look up per node."""
    dev = patch_preds.device
    acc = torch.zeros(GRID_H, GRID_W, device=dev)
    cnt = torch.zeros(GRID_H, GRID_W, device=dev)
    for pi in range(len(positions)):
        ri, ci = int(positions[pi, 0]), int(positions[pi, 1])
        acc[ri : ri + PATCH_H, ci : ci + PATCH_W] += patch_preds[pi]
        cnt[ri : ri + PATCH_H, ci : ci + PATCH_W] += 1.0
    pred_grid = acc / cnt.clamp(min=1)
    return pred_grid[node_grid_rc[:, 0], node_grid_rc[:, 1]]


@torch.no_grad()
def evaluate_pressure(
    model: PressureDQEModel,
    dataset: AirfRANSNearFieldDataset,
    device: torch.device,
    surf_weight: float = 10.0,
) -> dict[str, float]:
    """Surface-weighted MSE and R² for pressure, evaluated at node resolution."""
    model.eval()

    surf_se = vol_se = 0.0
    surf_n = vol_n = 0
    res_sq = true_sum = true_sq = 0.0
    total_n = 0

    for i in range(len(dataset)):
        gg = dataset[i]
        p_norm_node, smask, node_grid_rc = dataset.get_node_pressure_eval(i)

        patches_d = gg.patches.to(device)
        pred_patch, _, _, _ = model(patches_d)
        pred_patch_cpu = pred_patch.squeeze(-1).cpu()  # (P,)
        del patches_d, pred_patch

        pred_n = _reconstruct_node_predictions(pred_patch_cpu, gg.positions, node_grid_rc)

        # Use normalised values throughout — avoids p_std amplification blowing up MSE
        se = (pred_n - p_norm_node) ** 2
        surf_se += se[smask].sum().item()
        vol_se += se[~smask].sum().item()
        surf_n += int(smask.sum())
        vol_n += int((~smask).sum())
        res_sq += se.sum().item()
        true_sum += p_norm_node.sum().item()
        true_sq += (p_norm_node ** 2).sum().item()
        total_n += p_norm_node.shape[0]

    mse_w = (
        surf_se / max(surf_n, 1) * surf_weight + vol_se / max(vol_n, 1)
    ) / (surf_weight + 1)
    true_mean = true_sum / max(total_n, 1)
    true_var = max(true_sq / max(total_n, 1) - true_mean ** 2, 1e-12)
    r2 = 1.0 - res_sq / (total_n * true_var)

    return {"weighted_mse": float(mse_w), "r2_pressure": float(r2)}


# ── Embedding collection ──────────────────────────────────────────────────────


@torch.no_grad()
def collect_airfoil_embeddings(
    model: PressureDQEModel,
    dataset: AirfRANSNearFieldDataset,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (A, n_intervals) interval histograms and (A,) mean surface pressure per airfoil.

    Each airfoil is represented by the mean interval-assignment vector across
    its P patches — a pressure-quantile histogram in the DQE's learned interval
    basis.  This preserves the distributional shape (suction peak location,
    trailing-edge pressure recovery, upper/lower asymmetry) that mean-pooling
    raw embeddings would discard.

    Surface patches (patch_surf==True) determine the mean pressure used as the
    z-coordinate in the 3-D visualisation; falls back to all-patch mean if no
    surface patch exists.
    """
    model.eval()
    representations: list[torch.Tensor] = []
    mean_pressures: list[float] = []

    for i in range(len(dataset)):
        gg = dataset[i]
        _, _, iv_w, _ = model(gg.patches.to(device))  # iv_w: (P, n_intervals)
        representations.append(iv_w.mean(0).cpu())    # (n_intervals,) — quantile histogram

        smask = gg.patch_surf
        p_vals = gg.patch_pressure
        mean_pressures.append(
            p_vals[smask].mean().item() if smask.any() else p_vals.mean().item()
        )
        del iv_w

    return torch.stack(representations), torch.tensor(mean_pressures)


# ── Visualisation ─────────────────────────────────────────────────────────────


def visualize_pressure_surface(
    embeddings: torch.Tensor,      # (A, d)
    mean_pressures: torch.Tensor,  # (A,)  normalised
    p_mean: float,
    p_std: float,
    out_path: Path,
) -> None:
    """3-D pressure-similarity surface.

    Axes
    ----
    x, y  —  first two principal components of the per-airfoil embedding space
             (proximity = similar pressure-distribution shape)
    z     —  de-normalised mean surface pressure coefficient Cp

    A surface fitted via cubic griddata over the airfoil scatter forms a
    "Cp landscape": ridges and valleys reveal which regions of airfoil-design
    space tend to have high or low surface pressure.
    """
    E = embeddings.numpy().astype(np.float64)

    # PCA to 2D: centre → SVD → project onto first two right-singular vectors
    E_c = E - E.mean(0)
    _, _, Vt = np.linalg.svd(E_c, full_matrices=False)
    coords = E_c @ Vt[:2].T  # (A, 2)

    cp = mean_pressures.numpy() * p_std + p_mean  # de-normalise

    x, y, z = coords[:, 0], coords[:, 1], cp

    # Interpolated surface over the scatter.
    # "linear" avoids the oscillation artifacts and NaN interior regions that
    # "cubic" produces on sparse irregular scatter.  NaN outside the convex
    # hull is filled with the global mean so plot_surface has no holes.
    xi = np.linspace(x.min(), x.max(), 60)
    yi = np.linspace(y.min(), y.max(), 60)
    Xi, Yi = np.meshgrid(xi, yi)
    Zi = griddata((x, y), z, (Xi, Yi), method="linear")
    Zi = np.where(np.isnan(Zi), np.nanmean(z), Zi)

    fig = plt.figure(figsize=(13, 9))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot_surface(
        Xi, Yi, Zi,
        cmap="RdBu_r", alpha=0.50,
        linewidth=0, antialiased=True,
    )
    sc = ax.scatter(
        x, y, z,
        c=z, cmap="RdBu_r",
        s=40, depthshade=True,
        edgecolors="k", linewidths=0.3,
        zorder=5,
    )
    fig.colorbar(sc, ax=ax, shrink=0.5, pad=0.1, label="Mean surface Cp")

    ax.set_xlabel("PC 1  (embedding similarity)")
    ax.set_ylabel("PC 2")
    ax.set_zlabel("Mean surface Cp")
    ax.set_title(
        "Airfoil pressure-similarity surface\n"
        "Proximity on x-y plane → similar pressure distribution  ·  "
        "Height → mean surface Cp"
    )

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


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

    # ── Near-field domain ──────────────────────────────────────────────────
    # Update the module-level dict before any dataset or ReferenceDomain is
    # constructed — both reference NEAR_FIELD_BOUNDS by name at call time.
    NEAR_FIELD_BOUNDS["y_min"] = -args.near_field_y
    NEAR_FIELD_BOUNDS["y_max"] = args.near_field_y

    # ── Load raw data ──────────────────────────────────────────────────────
    print(f"Loading AirfRANS (task={args.task}) …")
    raw_train = AirfRANS(root=args.data_root, task=args.task, train=True)
    raw_test = AirfRANS(root=args.data_root, task=args.task, train=False)

    n_total = len(raw_train)
    val_size = max(1, int(n_total * 0.1))
    train_indices = list(range(n_total - val_size))
    val_indices = list(range(n_total - val_size, n_total))

    # ── Normalisation ──────────────────────────────────────────────────────
    print("  Computing normalisation stats …")
    x_mean, x_std, p_mean, p_std = compute_normalisation(raw_train, train_indices)
    norm_stats = (x_mean, x_std, p_mean, p_std)
    print(f"  Pressure  mean={p_mean.item():.4f}  std={p_std.item():.4f}")

    # ── Build datasets ─────────────────────────────────────────────────────
    # train and val share raw_train + norm_stats → one shared cache
    train_val_cache: dict[int, GraphGrid] = {}
    train_ds = AirfRANSNearFieldDataset(
        args.data_root, "train", task=args.task,
        norm_stats=norm_stats, indices=train_indices,
        _shared_raw=raw_train, _shared_cache=train_val_cache,
    )
    val_ds = AirfRANSNearFieldDataset(
        args.data_root, "train", task=args.task,
        norm_stats=norm_stats, indices=val_indices,
        _shared_raw=raw_train, _shared_cache=train_val_cache,
    )
    test_ds = AirfRANSNearFieldDataset(
        args.data_root, "test", task=args.task,
        norm_stats=norm_stats, _shared_raw=raw_test,
    )
    print(
        f"  {len(train_ds)} train / {len(val_ds)} val / {len(test_ds)} test\n"
        f"  Domain: x∈[{NEAR_FIELD_BOUNDS['x_min']}, {NEAR_FIELD_BOUNDS['x_max']}]  "
        f"y∈[{NEAR_FIELD_BOUNDS['y_min']}, {NEAR_FIELD_BOUNDS['y_max']}]\n"
        f"  Grid {GRID_H}×{GRID_W}  Patches/graph: {N_PATCHES} "
        f"({PATCH_H}×{PATCH_W} stride {PATCH_STRIDE})"
    )

    # ── Model ──────────────────────────────────────────────────────────────
    model = PressureDQEModel(
        in_channels=IN_CHANNELS,
        hidden_dim=args.hidden_dim,
        embed_dim=args.embed_dim,
        n_intervals=args.n_intervals,
        n_protos=args.n_protos,
        dropout=args.dropout,
        K_max=args.K_max,
        interval_temp=args.interval_temp,
        proto_temp=args.proto_temp,
    ).to(device)

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        print(f"Resumed from {args.resume}")

    p_mean_d = p_mean.to(device)
    p_std_d = p_std.to(device)

    history: dict[str, list[float]] = {
        "total": [], "regression": [], "centripetal": [], "spread": [],
    }

    # ── Phase 1: Warmup ────────────────────────────────────────────────────
    print(f"\n=== Phase 1: Warmup ({args.warmup_epochs} epochs) ===")
    model.set_phase(1)
    opt1 = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=1e-4,
    )
    sched1 = CosineAnnealingLR(opt1, T_max=max(1, args.warmup_epochs))

    for epoch in range(args.warmup_epochs):
        model.train()
        ep_loss = 0.0
        n_graphs = 0
        train_ds.shuffle()

        for gg in train_ds:
            patches_d = gg.patches.to(device)
            targets_d = gg.patch_pressure.unsqueeze(-1).to(device)  # (P, 1)

            opt1.zero_grad()
            pred = model.warmup_forward(patches_d)
            loss = F.mse_loss(pred, targets_d)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt1.step()

            ep_loss += loss.item()
            n_graphs += 1
            del patches_d, targets_d, pred

        sched1.step()
        if (epoch + 1) % max(1, args.warmup_epochs // 5) == 0:
            print(
                f"  [warmup] epoch {epoch + 1:3d}  "
                f"loss={ep_loss / max(1, n_graphs):.4f}"
            )

    # ── Phase 2: Full DQE training ─────────────────────────────────────────
    print(f"\n=== Phase 2: Full DQE training ({args.full_epochs} epochs) ===")
    model.set_phase(2)
    opt2 = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=1e-4,
    )
    sched2 = CosineAnnealingLR(opt2, T_max=max(1, args.full_epochs))

    best_val_mse = math.inf
    best_epoch = 0

    for epoch in range(args.full_epochs):
        model.train()
        ep_reg = ep_cent = 0.0
        n_graphs = 0
        train_ds.shuffle()

        ep_spread = 0.0
        for gg in train_ds:
            patches_d = gg.patches.to(device)
            targets_d = gg.patch_pressure.unsqueeze(-1).to(device)  # (P, 1)
            surf_d = gg.patch_surf.to(device)                        # (P,)

            opt2.zero_grad()
            pred, emb, _, pr_w = model(patches_d)

            # Surface-weighted pressure MSE
            per_patch = F.mse_loss(pred, targets_d, reduction="none").squeeze(-1)
            w = torch.where(
                surf_d,
                torch.full_like(per_patch, args.surf_weight),
                torch.ones_like(per_patch),
            )
            l_reg = (w * per_patch).sum() / w.sum()

            # Centripetal: pull each patch embedding toward its expected DQE prototype.
            # Detach pr_w to break the double-gradient path through protos_tan — the
            # assignment weights act as a fixed target here, not a learnable gate.
            expected_proto = pr_w.detach() @ model.dqe.protos_tan  # (P, d)
            l_cent = F.mse_loss(emb, expected_proto)

            # Spread: repel prototypes from each other so they can't collapse to the mean.
            # Hinge penalty fires when any pair is closer than proto_margin.
            P_mat = model.dqe.protos_tan                           # (n_protos, d)
            pdist = torch.cdist(P_mat, P_mat)                      # (n_protos, n_protos)
            off_diag = ~torch.eye(P_mat.shape[0], dtype=torch.bool, device=P_mat.device)
            l_spread = F.relu(args.proto_margin - pdist[off_diag]).mean()

            loss = l_reg + args.centripetal_weight * l_cent + args.spread_weight * l_spread
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt2.step()

            ep_reg += l_reg.item()
            ep_cent += l_cent.item()
            ep_spread += l_spread.item()
            n_graphs += 1
            del patches_d, targets_d, surf_d, pred, emb, pr_w

        sched2.step()
        ng = max(1, n_graphs)
        history["regression"].append(ep_reg / ng)
        history["centripetal"].append(ep_cent / ng)
        history["spread"].append(ep_spread / ng)
        history["total"].append(
            (ep_reg + args.centripetal_weight * ep_cent + args.spread_weight * ep_spread) / ng
        )

        if (epoch + 1) % max(1, args.full_epochs // 10) == 0:
            torch.cuda.empty_cache()
            val_m = evaluate_pressure(model, val_ds, device, args.surf_weight)
            kappa = model.dqe.kappa.item()
            print(
                f"  [full] epoch {epoch + 1:3d}  "
                f"reg={history['regression'][-1]:.4f}  "
                f"cent={history['centripetal'][-1]:.4f}  "
                f"val_mse={val_m['weighted_mse']:.4f}  "
                f"val_r2={val_m['r2_pressure']:.3f}  "
                f"kappa={kappa:.3f}"
            )
            if val_m["weighted_mse"] < best_val_mse:
                best_val_mse = val_m["weighted_mse"]
                best_epoch = epoch + 1
                torch.save(
                    {
                        "model": model.state_dict(),
                        "epoch": epoch + 1,
                        "val_metrics": val_m,
                    },
                    out_dir / "checkpoint.pt",
                )

    print(f"\nBest val MSE = {best_val_mse:.4f} at epoch {best_epoch}")

    # ── Final test evaluation ──────────────────────────────────────────────
    print("\n=== Final evaluation on test set ===")
    ckpt = torch.load(out_dir / "checkpoint.pt", map_location=device)
    model.load_state_dict(ckpt["model"])
    torch.cuda.empty_cache()

    test_m = evaluate_pressure(model, test_ds, device, args.surf_weight)
    for k, v in test_m.items():
        print(f"  {k}: {v:.4f}")

    with open(out_dir / "test_metrics.json", "w") as f:
        json.dump(test_m, f, indent=2)

    # ── Collect embeddings & visualise ─────────────────────────────────────
    print("\n=== Building 3-D pressure-similarity surface ===")
    torch.cuda.empty_cache()

    # Embed all splits; combine for a denser surface (more airfoils = cleaner landscape)
    embs_train, pvals_train = collect_airfoil_embeddings(model, train_ds, device)
    embs_val, pvals_val = collect_airfoil_embeddings(model, val_ds, device)
    embs_test, pvals_test = collect_airfoil_embeddings(model, test_ds, device)

    all_embs = torch.cat([embs_train, embs_val, embs_test], dim=0)
    all_pvals = torch.cat([pvals_train, pvals_val, pvals_test], dim=0)

    print(f"  {all_embs.shape[0]} airfoils  embed_dim={all_embs.shape[1]}")

    visualize_pressure_surface(
        all_embs, all_pvals,
        p_mean=p_mean.item(), p_std=p_std.item(),
        out_path=out_dir / "pressure_surface_3d.png",
    )

    # Training curves
    fig, axes = plt.subplots(1, 4, figsize=(20, 4))
    for ax, key in zip(axes, ["total", "regression", "centripetal", "spread"]):
        ax.plot(history[key])
        ax.set_title(key)
        ax.set_xlabel("epoch (phase 2)")
        ax.set_ylabel("loss")
    plt.tight_layout()
    plt.savefig(out_dir / "training_curves.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_dir / 'training_curves.png'}")

    print(f"\nAll outputs saved to {out_dir}")


# ── CLI ───────────────────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Pressure-DQE on near-airfoil patches → 3-D similarity surface",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Data
    p.add_argument("--data_root", default="../data/AirfRANS")
    p.add_argument("--task", default="scarce", choices=["scarce", "full"])
    p.add_argument(
        "--near_field_y", type=float, default=0.6,
        help="Half-height of the near-field domain (y ∈ [-near_field_y, +near_field_y]). "
             "AirfRANS 'scarce' reaches ~20° AoA; set ≥0.6 to avoid zero-fill corruption "
             "in high-AoA wake patches.",
    )

    # Model
    p.add_argument("--hidden_dim", type=int, default=128)
    p.add_argument("--embed_dim", type=int, default=64)
    p.add_argument("--n_intervals", type=int, default=8)
    p.add_argument("--n_protos", type=int, default=8)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--K_max", type=float, default=2.0)
    p.add_argument("--interval_temp", type=float, default=0.5)
    p.add_argument("--proto_temp", type=float, default=1.0)

    # Training
    p.add_argument("--warmup_epochs", type=int, default=30)
    p.add_argument("--full_epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=3e-4)

    # Loss
    p.add_argument("--surf_weight", type=float, default=10.0,
                   help="Surface patch loss multiplier")
    p.add_argument("--centripetal_weight", type=float, default=0.01,
                   help="Weight on centripetal (embed→prototype) regulariser")
    p.add_argument("--proto_margin", type=float, default=1.0,
                   help="Minimum L2 distance enforced between prototype pairs")
    p.add_argument("--spread_weight", type=float, default=0.1,
                   help="Weight on prototype spread (anti-collapse) regulariser")

    # Misc
    p.add_argument("--out_dir", default="results/pressure_dqe")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--resume", default="", help="Path to checkpoint to resume from")

    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    t0 = time.time()
    train(args)
    print(f"\nTotal wall time: {(time.time() - t0) / 60:.1f} min")
