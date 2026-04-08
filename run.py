"""
run.py — Joint Pressure+Shape DQE on near-airfoil patches → 3-D similarity surface.

Differences from run_airfrans_cnn.py
--------------------------------------
Near-field filtering
    The interpolation domain is clipped to a tight box around the airfoil
    (NEAR_FIELD_BOUNDS, default x∈[-0.25,1.5] y∈[-0.4,0.4]).  This gives
    roughly 4× better spatial resolution per grid cell compared with the
    full-domain bounds, so patches resolve boundary-layer and near-wake
    physics rather than far-field quiescent flow.

Joint pressure + shape lens
    Two DQE heads share a single CNN encoder:
      - pressure_dqe: predicts mean patch pressure; intervals/prototypes
        decompose the Cp distribution (suction peak, recovery, wake).
      - shape_dqe: predicts mean patch SDF (signed-distance-to-surface);
        intervals/prototypes decompose the airfoil geometry proximity,
        capturing leading-edge radius, camber, and thickness distribution.
    The shared encoder is forced to produce embeddings that are informative
    for both tasks simultaneously.  Loss is weighted sum with --shape_weight.

No stratification
    StratumAssigner and k-means phase are removed.  Two StratumDQE heads
    train directly on their targets with two phases:
      1. Warmup  — encoder + two linear regression heads (MSE on p and SDF)
      2. Full    — encoder + pressure_dqe + shape_dqe
                   loss = surf-weighted p MSE + shape_weight × SDF MSE
                          + centripetal + spread (for both DQE heads)

3-D surface visualisation
    After training, each airfoil is represented by the concatenated mean
    interval-assignment vectors from both DQE heads, giving a joint
    pressure-and-shape descriptor.  PCA projects to 2-D for the x-y plane;
    mean surface pressure provides the z-axis.  Proximity on the x-y plane
    reflects similarity in *both* aerodynamic loading and geometric shape.

Usage
-----
  python run.py --out_dir results/pressure_dqe

  # Quick smoke test
  python run.py --n_intervals 4 --n_protos 4 --warmup_epochs 5 --full_epochs 10

  # Pressure-only (shape_weight=0 recovers original behaviour)
  python run.py --shape_weight 0.0
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from scipy.interpolate import griddata
from scipy.stats import pearsonr
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
SDF_IDX: int = 4       # SDF channel in x_aug = [x, y, u_x_inf, u_y_inf, SDF, nx, ny]

N_CHORD: int = 128     # chord-wise interpolation points for shape vectors


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
    """CNN encoder + dual DQE heads predicting pressure distribution and airfoil shape.

    Two StratumDQE heads share a single CNN encoder:
      - pressure_dqe: predicts mean patch Cp; intervals/prototypes decompose
        the pressure distribution (suction peak, recovery, wake signature).
      - shape_dqe: predicts mean patch SDF (signed-distance-to-surface);
        intervals/prototypes decompose geometric proximity to the surface,
        capturing leading-edge radius, camber, and thickness.

    Training phases:
      Phase 1 — encoder + two WarmupRegressors (MSE on pressure and SDF)
      Phase 2 — full model: encoder + pressure_dqe + shape_dqe
                loss = surf-weighted pressure MSE
                       + shape_weight × surf-weighted SDF MSE
                       + centripetal + spread (both DQE heads)

    Post-training, ``collect_airfoil_embeddings`` concatenates the mean
    interval-assignment vectors from both heads into a joint descriptor that
    encodes *both* aerodynamic loading and geometric shape.
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
        self.pressure_dqe = StratumDQE(
            embed_dim, n_intervals, n_protos,
            n_targets=1,
            K_max=K_max,
            interval_temp=interval_temp,
            proto_temp=proto_temp,
        )
        self.shape_dqe = StratumDQE(
            embed_dim, n_intervals, n_protos,
            n_targets=1,
            K_max=K_max,
            interval_temp=interval_temp,
            proto_temp=proto_temp,
        )
        self.warmup_head = WarmupRegressor(embed_dim, n_targets=1)
        self.warmup_shape_head = WarmupRegressor(embed_dim, n_targets=1)

    def set_phase(self, phase: int) -> None:
        for p in self.parameters():
            p.requires_grad_(False)
        if phase == 1:
            for p in self.encoder.parameters():
                p.requires_grad_(True)
            for p in self.warmup_head.parameters():
                p.requires_grad_(True)
            for p in self.warmup_shape_head.parameters():
                p.requires_grad_(True)
        elif phase == 2:
            for p in self.parameters():
                p.requires_grad_(True)

    def encode(self, patches: torch.Tensor) -> torch.Tensor:
        return self.encoder(patches)

    def warmup_forward(
        self, patches: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (pred_pressure (P,1), pred_sdf (P,1))."""
        emb = self.encode(patches)
        return self.warmup_head(emb), self.warmup_shape_head(emb)

    def forward(
        self, patches: torch.Tensor
    ) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor,
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
    ]:
        """Returns (p_pred, s_pred, emb, p_iv_w, p_pr_w, s_iv_w, s_pr_w).

        p_pred  : (P, 1) predicted pressure
        s_pred  : (P, 1) predicted SDF (shape)
        emb     : (P, d) shared encoder embeddings
        p_iv_w  : (P, I) pressure interval-assignment weights
        p_pr_w  : (P, K) pressure prototype-assignment weights
        s_iv_w  : (P, I) shape interval-assignment weights
        s_pr_w  : (P, K) shape prototype-assignment weights
        """
        emb = self.encode(patches)
        p_pred, p_iv_w, p_pr_w = self.pressure_dqe(emb)
        s_pred, s_iv_w, s_pr_w = self.shape_dqe(emb)
        return p_pred, s_pred, emb, p_iv_w, p_pr_w, s_iv_w, s_pr_w


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

    for i in tqdm(range(len(dataset)), desc="evaluate", leave=False, unit="graph"):
        gg = dataset[i]
        p_norm_node, smask, node_grid_rc = dataset.get_node_pressure_eval(i)

        patches_d = gg.patches.to(device)
        pred_patch, _, _, _, _, _, _ = model(patches_d)
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
    """Return joint (A, 2*n_intervals) histograms and (A,) mean surface pressure per airfoil.

    Each airfoil is represented by the concatenation of:
      - mean pressure interval-assignment vector (pressure-quantile histogram)
      - mean shape interval-assignment vector (SDF-proximity histogram)

    This joint descriptor captures both the aerodynamic loading distribution
    (suction peak location, recovery, wake) and the geometric shape (leading-
    edge radius, camber, thickness) so that proximity in the embedding space
    reflects similarity in *both* pressure and geometry.

    Surface patches (patch_surf==True) determine the mean pressure used as the
    z-coordinate in the 3-D visualisation; falls back to all-patch mean if no
    surface patch exists.
    """
    model.eval()
    representations: list[torch.Tensor] = []
    mean_pressures: list[float] = []

    for i in tqdm(range(len(dataset)), desc="collect embeddings", leave=False, unit="graph"):
        gg = dataset[i]
        # p_iv_w: pressure quantile histogram; s_iv_w: shape/SDF histogram
        _, _, _, p_iv_w, _, s_iv_w, _ = model(gg.patches.to(device))
        joint = torch.cat([p_iv_w.mean(0), s_iv_w.mean(0)], dim=0).cpu()  # (2*n_intervals,)
        representations.append(joint)

        smask = gg.patch_surf
        p_vals = gg.patch_pressure
        mean_pressures.append(
            p_vals[smask].mean().item() if smask.any() else p_vals.mean().item()
        )
        del p_iv_w, s_iv_w

    return torch.stack(representations), torch.tensor(mean_pressures)


# ── Shape metric ──────────────────────────────────────────────────────────────


def compute_shape_vector(surf_pos: np.ndarray) -> np.ndarray:
    """Chord-wise [thickness(x), camber(x)] vector from airfoil surface nodes.

    Parameters
    ----------
    surf_pos : (N_surf, 2) — (x, y) coordinates of surface nodes in physical space

    Returns
    -------
    (2 * N_CHORD,) — stacked [thickness, camber] at N_CHORD chord stations,
    chord-normalised to x ∈ [0, 1].  Returns zeros when the surface has fewer
    than 10 nodes or a degenerate chord (< 1e-6).
    """
    if surf_pos.shape[0] < 10:
        return np.zeros(2 * N_CHORD)

    x, y = surf_pos[:, 0], surf_pos[:, 1]
    chord = x.max() - x.min()
    if chord < 1e-6:
        return np.zeros(2 * N_CHORD)
    x_n = (x - x.min()) / chord

    y_mid = y.mean()
    upper, lower = y >= y_mid, y < y_mid
    if upper.sum() < 6 or lower.sum() < 6:
        return np.zeros(2 * N_CHORD)

    x_grid = np.linspace(0.0, 1.0, N_CHORD)
    su = np.argsort(x_n[upper])
    y_up = np.interp(x_grid, x_n[upper][su], y[upper][su])
    sl = np.argsort(x_n[lower])
    y_lo = np.interp(x_grid, x_n[lower][sl], y[lower][sl])

    return np.concatenate([y_up - y_lo, (y_up + y_lo) / 2.0])


def collect_shape_vectors(
    raw_ds: AirfRANS,
    dataset: AirfRANSNearFieldDataset,
) -> np.ndarray:
    """Collect (A, 2*N_CHORD) shape vectors in dataset order using parallel workers.

    Surface positions are extracted in the main process (dataset access is not
    picklable), then compute_shape_vector is farmed out to a process pool.
    """
    surf_positions: list[np.ndarray] = []
    for i in tqdm(range(len(dataset)), desc="  surface nodes", leave=False, unit="g"):
        raw_idx = dataset._indices[dataset._order[i]]
        raw: Data = raw_ds[raw_idx]  # type: ignore[assignment]
        pos = cast(torch.Tensor, raw.pos).float().numpy()
        surf_mask = (
            cast(torch.Tensor, raw.surf).bool().numpy()
            if hasattr(raw, "surf") and raw.surf is not None
            else np.zeros(pos.shape[0], dtype=bool)
        )
        surf_positions.append(pos[surf_mask])

    with ProcessPoolExecutor() as pool:
        vecs = list(tqdm(
            pool.map(compute_shape_vector, surf_positions),
            total=len(surf_positions),
            desc="  shape vectors",
            leave=False,
            unit="g",
        ))
    return np.stack(vecs)


# ── Visualisation ─────────────────────────────────────────────────────────────


def visualize_shape_pressure(
    all_embs: np.ndarray,    # (A, d)   joint pressure+shape DQE embeddings
    all_pvals: np.ndarray,   # (A,)     normalised mean surface pressure
    all_shape: np.ndarray,   # (A, 2*N_CHORD) geometric shape vectors
    p_mean: float,
    p_std: float,
    out_path: Path,
) -> None:
    """Two-panel figure: joint embedding surface overlaid with shape, + shape space.

    Left (3-D)
        Joint embedding PCA (x/y = PC1/PC2, z = mean surface Cp).
        The surface mesh encodes mean Cp; scatter points are coloured by
        geometric shape PC1.  Proximity on the x-y plane reflects similarity
        in *both* pressure distribution and airfoil geometry (as learned by
        the dual DQE heads).  If shape-similar airfoils cluster together, the
        shape DQE head has successfully encoded geometry.

    Right (2-D)
        Geometric shape-PCA space (x/y = shape PC1/PC2), scatter coloured by
        mean Cp.  Comparing left and right reveals how well the model's joint
        embedding tracks geometric shape.

    The Pearson correlation between joint embedding PC1 and geometric shape
    PC1 is annotated on both panels.
    """
    cp = all_pvals * p_std + p_mean  # de-normalise Cp

    # PCA of joint embedding
    E_c = all_embs - all_embs.mean(0)
    _, _, Vt_e = np.linalg.svd(E_c, full_matrices=False)
    e_coords = E_c @ Vt_e[:2].T  # (A, 2)

    # PCA of geometric shape vectors
    S_c = all_shape - all_shape.mean(0)
    _, _, Vt_s = np.linalg.svd(S_c, full_matrices=False)
    s_coords = S_c @ Vt_s[:2].T  # (A, 2)

    r_val, p_val = pearsonr(e_coords[:, 0], s_coords[:, 0])
    print(f"  Embedding PC1 vs Geometric shape PC1:  r = {r_val:+.3f}  (p = {p_val:.2e})")

    xp, yp = e_coords[:, 0], e_coords[:, 1]
    xs, ys = s_coords[:, 0], s_coords[:, 1]
    shape_c = s_coords[:, 0]  # colour by geometric shape PC1
    cp_min, cp_max = float(cp.min()), float(cp.max())

    # Interpolate Cp surface over the embedding scatter
    xi = np.linspace(xp.min(), xp.max(), 60)
    yi = np.linspace(yp.min(), yp.max(), 60)
    Xi, Yi = np.meshgrid(xi, yi)
    Zi = griddata((xp, yp), cp, (Xi, Yi), method="linear")
    Zi = np.where(np.isnan(Zi), np.nanmean(cp), Zi)

    fig = plt.figure(figsize=(19, 8))
    gs = gridspec.GridSpec(
        1, 2,
        width_ratios=[1.45, 1],
        wspace=0.08, left=0.04, right=0.97, top=0.93, bottom=0.07,
    )

    # ── Left: 3-D embedding surface coloured by geometric shape ───────────
    ax3 = fig.add_subplot(gs[0], projection="3d")
    ax3.plot_surface(
        Xi, Yi, Zi,
        cmap="RdBu_r", alpha=0.45, linewidth=0, antialiased=True,
        vmin=cp_min, vmax=cp_max,
    )
    sc_shape = ax3.scatter(
        xp, yp, zs=cp,
        c=shape_c, cmap="viridis",
        vmin=float(shape_c.min()), vmax=float(shape_c.max()),
        s=45, depthshade=True, edgecolors="k", linewidths=0.3, zorder=5,
    )
    fig.colorbar(sc_shape, ax=ax3, shrink=0.42, pad=0.04,
                 label="Geometric shape PC 1  (thickness/camber mode)")
    ax3.set_xlabel("Embedding PC 1", labelpad=6)
    ax3.set_ylabel("Embedding PC 2", labelpad=6)
    ax3.set_zlabel("Mean surface Cp", labelpad=6)
    ax3.set_title(
        "Joint pressure+shape embedding  ·  colour = geometric shape\n"
        f"Embedding–geometry correlation:  r = {r_val:+.3f}",
        fontsize=10, pad=10,
    )

    # ── Right: 2-D geometric shape space coloured by Cp ───────────────────
    ax2 = fig.add_subplot(gs[1])
    sc_cp = ax2.scatter(
        xs, ys, c=cp, cmap="RdBu_r",
        vmin=cp_min, vmax=cp_max,
        s=42, edgecolors="k", linewidths=0.3, alpha=0.88,
    )
    fig.colorbar(sc_cp, ax=ax2, shrink=0.78, label="Mean surface Cp")
    ax2.set_xlabel("Geometric shape PC 1  (thickness/camber mode)")
    ax2.set_ylabel("Geometric shape PC 2")
    ax2.set_title("Geometric shape similarity", fontsize=10)
    ax2.set_aspect("equal", adjustable="datalim")
    ax2.grid(True, lw=0.4, alpha=0.4)
    ax2.annotate(
        f"Embedding–geometry\ncorrelation\nr = {r_val:+.3f}",
        xy=(0.04, 0.96), xycoords="axes fraction",
        va="top", ha="left", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#999", alpha=0.85),
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
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

    history: dict[str, list[float]] = {
        "total": [], "p_regression": [], "s_regression": [], "centripetal": [], "spread": [],
    }

    # ── Phase 1: Warmup ────────────────────────────────────────────────────
    print(f"\n=== Phase 1: Warmup ({args.warmup_epochs} epochs) ===")
    model.set_phase(1)
    opt1 = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=1e-4,
    )
    sched1 = CosineAnnealingLR(opt1, T_max=max(1, args.warmup_epochs))

    epoch_bar1 = tqdm(range(args.warmup_epochs), desc="warmup", unit="ep")
    for epoch in epoch_bar1:
        model.train()
        ep_loss = 0.0
        n_graphs = 0
        train_ds.shuffle()

        for gg in tqdm(train_ds, desc="  graphs", leave=False, unit="g"):
            patches_d = gg.patches.to(device)
            p_targets_d = gg.patch_pressure.unsqueeze(-1).to(device)        # (P, 1)
            s_targets_d = patches_d[:, SDF_IDX].mean(dim=(-1, -2)).unsqueeze(-1)  # (P, 1)

            opt1.zero_grad()
            p_pred, s_pred = model.warmup_forward(patches_d)
            loss = F.mse_loss(p_pred, p_targets_d) + args.shape_weight * F.mse_loss(s_pred, s_targets_d)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt1.step()

            ep_loss += loss.item()
            n_graphs += 1
            del patches_d, p_targets_d, s_targets_d, p_pred, s_pred

        sched1.step()
        epoch_bar1.set_postfix(loss=f"{ep_loss / max(1, n_graphs):.4f}")

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

    def _spread_loss(protos: torch.Tensor, margin: float) -> torch.Tensor:
        """Hinge penalty repelling prototype pairs closer than margin."""
        pdist = torch.cdist(protos, protos)
        off = ~torch.eye(protos.shape[0], dtype=torch.bool, device=protos.device)
        return F.relu(margin - pdist[off]).mean()

    epoch_bar2 = tqdm(range(args.full_epochs), desc="full DQE", unit="ep")
    for epoch in epoch_bar2:
        model.train()
        ep_p_reg = ep_s_reg = ep_cent = ep_spread = 0.0
        n_graphs = 0
        train_ds.shuffle()
        for gg in tqdm(train_ds, desc="  graphs", leave=False, unit="g"):
            patches_d = gg.patches.to(device)
            p_targets_d = gg.patch_pressure.unsqueeze(-1).to(device)        # (P, 1)
            s_targets_d = patches_d[:, SDF_IDX].mean(dim=(-1, -2)).unsqueeze(-1)  # (P, 1)
            surf_d = gg.patch_surf.to(device)                               # (P,)

            opt2.zero_grad()
            p_pred, s_pred, emb, _, p_pr_w, _, s_pr_w = model(patches_d)

            # Surface-weighted loss weights (same for both heads)
            per_p = F.mse_loss(p_pred, p_targets_d, reduction="none").squeeze(-1)
            per_s = F.mse_loss(s_pred, s_targets_d, reduction="none").squeeze(-1)
            w = torch.where(
                surf_d,
                torch.full_like(per_p, args.surf_weight),
                torch.ones_like(per_p),
            )
            l_reg_p = (w * per_p).sum() / w.sum()
            l_reg_s = (w * per_s).sum() / w.sum()

            # Centripetal: pull each patch embedding toward its expected prototype.
            # Detach assignment weights to break the double-gradient through protos_tan.
            p_expected = p_pr_w.detach() @ model.pressure_dqe.protos_tan  # (P, d)
            s_expected = s_pr_w.detach() @ model.shape_dqe.protos_tan     # (P, d)
            # Average centripetal over both heads so their gradients compete fairly
            l_cent = 0.5 * (F.mse_loss(emb, p_expected) + F.mse_loss(emb, s_expected))

            # Spread: repel prototypes for each head independently
            l_spread = 0.5 * (
                _spread_loss(model.pressure_dqe.protos_tan, args.proto_margin)
                + _spread_loss(model.shape_dqe.protos_tan, args.proto_margin)
            )

            loss = (
                l_reg_p + args.shape_weight * l_reg_s
                + args.centripetal_weight * l_cent
                + args.spread_weight * l_spread
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt2.step()

            ep_p_reg += l_reg_p.item()
            ep_s_reg += l_reg_s.item()
            ep_cent += l_cent.item()
            ep_spread += l_spread.item()
            n_graphs += 1
            del patches_d, p_targets_d, s_targets_d, surf_d, p_pred, s_pred, emb, p_pr_w, s_pr_w

        sched2.step()
        ng = max(1, n_graphs)
        history["p_regression"].append(ep_p_reg / ng)
        history["s_regression"].append(ep_s_reg / ng)
        history["centripetal"].append(ep_cent / ng)
        history["spread"].append(ep_spread / ng)
        history["total"].append(
            (ep_p_reg + args.shape_weight * ep_s_reg
             + args.centripetal_weight * ep_cent
             + args.spread_weight * ep_spread) / ng
        )

        epoch_bar2.set_postfix(
            p_reg=f"{history['p_regression'][-1]:.4f}",
            s_reg=f"{history['s_regression'][-1]:.4f}",
            cent=f"{history['centripetal'][-1]:.4f}",
        )

        if (epoch + 1) % max(1, args.full_epochs // 10) == 0:
            torch.cuda.empty_cache()
            val_m = evaluate_pressure(model, val_ds, device, args.surf_weight)
            kappa_p = model.pressure_dqe.kappa.item()
            kappa_s = model.shape_dqe.kappa.item()
            epoch_bar2.write(
                f"  epoch {epoch + 1:3d}  "
                f"val_mse={val_m['weighted_mse']:.4f}  "
                f"val_r2={val_m['r2_pressure']:.3f}  "
                f"κ_p={kappa_p:.3f}  κ_s={kappa_s:.3f}"
            )
            if val_m["weighted_mse"] < best_val_mse:
                best_val_mse = val_m["weighted_mse"]
                best_epoch = epoch + 1
                torch.save(
                    {
                        "model": model.state_dict(),
                        "epoch": epoch + 1,
                        "val_metrics": val_m,
                        "norm_stats": {
                            "x_mean": x_mean.cpu(),
                            "x_std": x_std.cpu(),
                            "p_mean": p_mean.cpu(),
                            "p_std": p_std.cpu(),
                        },
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
    print("\n=== Collecting embeddings and building visualisations ===")
    torch.cuda.empty_cache()

    # Joint DQE embeddings — combine all splits for a denser surface
    embs_train, pvals_train = collect_airfoil_embeddings(model, train_ds, device)
    embs_val, pvals_val = collect_airfoil_embeddings(model, val_ds, device)
    embs_test, pvals_test = collect_airfoil_embeddings(model, test_ds, device)

    all_embs = torch.cat([embs_train, embs_val, embs_test], dim=0).numpy().astype(np.float64)
    all_pvals = torch.cat([pvals_train, pvals_val, pvals_test], dim=0).numpy()

    print(f"  {all_embs.shape[0]} airfoils  embed_dim={all_embs.shape[1]}")

    # Geometric shape vectors (chord-wise thickness/camber from surface nodes)
    print("  Computing geometric shape vectors …")
    shape_tr = collect_shape_vectors(raw_train, train_ds)
    shape_va = collect_shape_vectors(raw_train, val_ds)
    shape_te = collect_shape_vectors(raw_test, test_ds)
    all_shape = np.concatenate([shape_tr, shape_va, shape_te])

    visualize_shape_pressure(
        all_embs, all_pvals, all_shape,
        p_mean=p_mean.item(), p_std=p_std.item(),
        out_path=out_dir / "shape_pressure_surface.png",
    )

    # Training curves
    fig, axes = plt.subplots(1, 5, figsize=(25, 4))
    for ax, key in zip(axes, ["total", "p_regression", "s_regression", "centripetal", "spread"]):
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
    p.add_argument("--shape_weight", type=float, default=1.0,
                   help="Weight on shape (SDF) task relative to pressure task; 0 = pressure-only")
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
