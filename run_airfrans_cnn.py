"""
run_airfrans_cnn.py — Train StratifiedDQE on AirfRANS using a CNN patch encoder.

Instead of building a KNN edge-index and running GraphSAGE, node features are
interpolated from the ~180 k-node point cloud onto a regular (H × W) grid.
Image patches are then extracted from this grid and fed to a CNN encoder that
produces per-patch embeddings.  The rest of the StratifiedDQE pipeline —
stratum assigner, per-stratum DQE heads with learnable Riemannian curvature,
and all four training phases — is identical to run_airfrans.py.

Memory profile
--------------
No edge_index, no NeighborLoader.
One interpolated grid ≈ 730 KB (128 × 128 × 11 channels × float32).
200 training graphs ≈ 146 MB cached grid storage.
Each graph produces ≈ 49 patches (32 × 32, stride 16); these form the
natural mini-batch — no large tensor accumulation.

Usage examples
--------------
  python run_airfrans_cnn.py --out_dir results/airfrans_cnn

  # Quick smoke test
  python run_airfrans_cnn.py --n_strata 2 --embed_dim 32 --warmup_epochs 5

Training phases
---------------
1. Warmup regressor   (--warmup_epochs)
2. K-means cluster    (once after warmup)
3. Stratum assigner   (--assigner_epochs)
4. Full joint train   (--full_epochs)
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import pathlib
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.datasets import AirfRANS
from torch_geometric.data import Data

# ── AdaptiveCFD geometry utilities ───────────────────────────────────────────
from models.GeometricCNNAutoencoder import (
    GeometricPatchExtractor,
    ReferenceDomain,
)

from StratifiedEmbedding import (
    StratumAssigner,
    StratumDQE,
    total_loss,
    prediction_diversity_loss,
    conditional_diversity_loss,
    plot_training_curves,
    plot_curvature_evolution,
    plot_geometry_summary,
)

# ── Constants ─────────────────────────────────────────────────────────────────

TARGET_NAMES = ["u_x", "u_y", "p", "nu_t"]
N_TARGETS = 4
BASE_IN_DIM = 5  # raw AirfRANS node features (excl. spatial pos)
IN_CHANNELS = 7  # pos(2) + x(5) concatenated

# Reference domain for all simulations — AirfRANS airfoils are roughly
# centred at the origin with unit chord; use the same bounds as AirfoilPatchDataset.
DOMAIN_BOUNDS: dict[str, float] = {
    "x_min": -1.0,
    "x_max": 2.0,
    "y_min": -1.5,
    "y_max": 1.5,
}
GRID_H: int = 128
GRID_W: int = 128
PATCH_H: int = 32
PATCH_W: int = 32
PATCH_STRIDE: int = 16

# Derived: number of patches per axis (same for H and W given square grid + patches)
N_PATCHES_H: int = (GRID_H - PATCH_H) // PATCH_STRIDE + 1  # 7
N_PATCHES_W: int = (GRID_W - PATCH_W) // PATCH_STRIDE + 1  # 7
N_PATCHES: int = N_PATCHES_H * N_PATCHES_W  # 49


# ── Grid / patch data container ───────────────────────────────────────────────


@dataclass
class GraphGrid:
    """Per-graph patch representation built from an interpolated flow grid.

    All tensors are on CPU; moved to device inside training loops.

    Attributes
    ----------
    patches       : (P, C, h, w) — CNN input patches, features already normalised
    patch_targets : (P, T)       — mean normalised target per patch
    patch_surf    : (P,)  bool   — True if any surface node falls in the patch region
    positions     : (P, 2) int   — (row, col) top-left corner of each patch in the grid
    node_pos      : (N, 2)       — original CFD node coordinates
    node_targets  : (N, T)       — original normalised targets (used for eval metrics)
    node_surf     : (N,)  bool   — surface mask from AirfRANS
    node_grid_rc  : (N, 2) int   — (row, col) grid indices for each node
    """

    patches: torch.Tensor
    patch_targets: torch.Tensor
    patch_surf: torch.Tensor
    positions: torch.Tensor
    node_pos: torch.Tensor
    node_targets: torch.Tensor
    node_surf: torch.Tensor
    node_grid_rc: torch.Tensor


# ── Dataset ───────────────────────────────────────────────────────────────────


class AirfRANSGridDataset:
    """Lazy per-sample loader that interpolates AirfRANS point clouds to a grid.

    On first access to graph *i*, the point cloud is mapped via scipy griddata
    to a (GRID_H × GRID_W) regular grid and the resulting ``GraphGrid`` is
    stored in a RAM cache.  Subsequent accesses reuse the cached object.

    A single ``AirfRANS`` instance can be shared across multiple views (train,
    val, normalisation) via the ``_shared_raw`` parameter to avoid duplicating
    the dataset in memory.

    Parameters
    ----------
    norm_stats : tuple (x_mean, x_std, y_mean, y_std) | None
        Per-feature normalisation statistics computed over the training set.
        When None, raw values are used (suitable for the normalisation pass).
    _shared_raw : AirfRANS | None
        Pre-loaded AirfRANS instance to share.  If None, one is created.
    """

    def __init__(
        self,
        root: str,
        split: str,
        task: str = "scarce",
        norm_stats: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        | None = None,
        indices: list[int] | None = None,
        _shared_raw: AirfRANS | None = None,
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

        # Interpolation infrastructure — shared across all accesses.
        self._ref_domain = ReferenceDomain(
            grid_size=(GRID_H, GRID_W), bounds=DOMAIN_BOUNDS
        )
        self._extractor = GeometricPatchExtractor(
            patch_size=(PATCH_H, PATCH_W), stride=PATCH_STRIDE
        )

        # Grid cache: raw_idx → GraphGrid (None = not yet computed)
        self._cache: dict[int, GraphGrid] = {}

    # ── public API ────────────────────────────────────────────────────────────

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

    # ── internals ─────────────────────────────────────────────────────────────

    def _build_grid(self, raw_idx: int) -> GraphGrid:
        """Interpolate one AirfRANS graph to a regular grid and extract patches."""
        raw: Data = self._raw[raw_idx]  # type: ignore[assignment]

        pos_ = cast(torch.Tensor, raw.pos).float()  # (N, 2)
        x_feat = cast(torch.Tensor, raw.x).float()  # (N, 5)
        y_raw = cast(torch.Tensor, raw.y).float()  # (N, T)
        surf = (
            cast(torch.Tensor, raw.surf).bool()
            if hasattr(raw, "surf") and raw.surf is not None
            else torch.zeros(pos_.shape[0], dtype=torch.bool)
        )

        x_aug = torch.cat([pos_, x_feat], dim=-1)  # (N, 7)

        # Apply normalisation in feature space before interpolation.
        # Linear interp of normalised values == normalise interp of raw values.
        if self.norm_stats is not None:
            x_mean, x_std, y_mean, y_std = self.norm_stats
            x_aug = (x_aug - x_mean) / x_std
            y_raw = (y_raw - y_mean) / y_std

        pos_np = pos_.numpy()  # (N, 2)
        x_np = x_aug.numpy()  # (N, 7)
        y_np = y_raw.numpy()  # (N, T)
        s_np = surf.float().numpy()  # (N,) float for interpolation

        # ── Interpolate to grid ────────────────────────────────────────────
        # interpolate_field handles multi-channel (N, C) inputs.
        grid_x = self._ref_domain.interpolate_field(pos_np, x_np)  # (H, W, 7)
        grid_y = self._ref_domain.interpolate_field(pos_np, y_np)  # (H, W, T)
        grid_s = self._ref_domain.interpolate_field(pos_np, s_np)  # (H, W)

        # ── Extract patches ───────────────────────────────────────────────
        flow_patches_np, _, positions_list, _ = (
            self._extractor.extract_patches_with_positions(grid_x)
        )
        # flow_patches_np: (P, h, w, C) numpy

        P = len(positions_list)
        positions_t = torch.tensor(positions_list, dtype=torch.long)  # (P, 2)

        # CNN input: (P, C, h, w)
        patches_t = torch.from_numpy(flow_patches_np).float().permute(0, 3, 1, 2)

        # Per-patch mean target and surf flag
        patch_targets = torch.zeros(P, N_TARGETS, dtype=torch.float32)
        patch_surf = torch.zeros(P, dtype=torch.bool)
        grid_y_t = torch.from_numpy(grid_y).float()  # (H, W, T)
        grid_s_t = torch.from_numpy(grid_s).float()  # (H, W)

        for pi, (ri, ci) in enumerate(positions_list):
            patch_targets[pi] = (
                grid_y_t[ri : ri + PATCH_H, ci : ci + PATCH_W]
                .reshape(-1, N_TARGETS)
                .mean(0)
            )
            patch_surf[pi] = grid_s_t[ri : ri + PATCH_H, ci : ci + PATCH_W].max() > 0.1

        # ── Node → grid index mapping ─────────────────────────────────────
        x_min, x_max = DOMAIN_BOUNDS["x_min"], DOMAIN_BOUNDS["x_max"]
        y_min, y_max = DOMAIN_BOUNDS["y_min"], DOMAIN_BOUNDS["y_max"]
        col_f = (pos_[:, 0] - x_min) / (x_max - x_min) * (GRID_W - 1)
        row_f = (pos_[:, 1] - y_min) / (y_max - y_min) * (GRID_H - 1)
        col_i = col_f.round().long().clamp(0, GRID_W - 1)
        row_i = row_f.round().long().clamp(0, GRID_H - 1)
        node_grid_rc = torch.stack([row_i, col_i], dim=-1)  # (N, 2)

        return GraphGrid(
            patches=patches_t,
            patch_targets=patch_targets,
            patch_surf=patch_surf,
            positions=positions_t,
            node_pos=pos_.clone(),
            node_targets=torch.from_numpy(y_np).float(),
            node_surf=surf.clone(),
            node_grid_rc=node_grid_rc,
        )


# ── Normalisation ─────────────────────────────────────────────────────────────


def compute_normalisation_streaming(
    dataset: AirfRANSGridDataset,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Streaming per-feature mean / std over all training graphs.

    Operates on the raw AirfRANS data (before interpolation) to stay
    consistent with run_airfrans.py.  The resulting stats are stored in
    the dataset wrapper and applied before interpolation.
    """
    raw_ds = dataset._raw
    indices = dataset._indices

    first_raw: Data = raw_ds[indices[0]]  # type: ignore[assignment]
    pos0 = cast(torch.Tensor, first_raw.pos).float()
    x0 = cast(torch.Tensor, first_raw.x).float()
    y0 = cast(torch.Tensor, first_raw.y).float()
    x_aug0 = torch.cat([pos0, x0], dim=-1)

    Fx = x_aug0.shape[1]
    Fy = y0.shape[1]

    x_sum = torch.zeros(Fx)
    x_sum2 = torch.zeros(Fx)
    y_sum = torch.zeros(Fy)
    y_sum2 = torch.zeros(Fy)
    n_x = n_y = 0

    for idx in indices:
        raw_data: Data = raw_ds[idx]  # type: ignore[assignment]
        pos_ = cast(torch.Tensor, raw_data.pos).float()
        x_ = cast(torch.Tensor, raw_data.x).float()
        y_ = cast(torch.Tensor, raw_data.y).float()
        x_aug = torch.cat([pos_, x_], dim=-1)

        x_sum += x_aug.sum(0)
        x_sum2 += (x_aug**2).sum(0)
        y_sum += y_.sum(0)
        y_sum2 += (y_**2).sum(0)
        n_x += x_aug.shape[0]
        n_y += y_.shape[0]

    x_mean = x_sum / n_x
    x_std = ((x_sum2 / n_x) - x_mean**2).clamp(min=0).sqrt().clamp(min=1e-6)
    y_mean = y_sum / n_y
    y_std = ((y_sum2 / n_y) - y_mean**2).clamp(min=0).sqrt().clamp(min=1e-6)
    return x_mean, x_std, y_mean, y_std


# ── CNN patch encoder ─────────────────────────────────────────────────────────


class PatchCNNEncoder(nn.Module):
    """Convolutional encoder mapping (B, C, h, w) image patches to (B, embed_dim).

    Architecture: three stride-2 convolutions (doubling channels each time)
    followed by AdaptiveAvgPool2d(1) and a linear projection.

    For 32 × 32 patches the feature map sequence is:
      32×32 → 16×16 → 8×8 → 4×4 → pooled to 1×1

    Batch-norm is replaced with LayerNorm over the channel axis so that the
    model behaves consistently at inference time regardless of batch size
    (single-patch batches arise during the k-means collection phase).
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
            nn.AdaptiveAvgPool2d(1),  # (B, embed_dim, 1, 1)
        )
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, C, h, w) → (B, embed_dim)
        h = self.convs(x).squeeze(-1).squeeze(-1)  # (B, embed_dim)
        return self.proj(self.dropout(h))


# ── Warmup head (local copy to avoid GNN import dependency) ──────────────────


class WarmupRegressor(nn.Module):
    def __init__(self, embed_dim: int, n_targets: int, hidden: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden), nn.SiLU(), nn.Linear(hidden, n_targets)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── StratifiedDQE with CNN encoder ───────────────────────────────────────────


class StratifiedDQECNN(nn.Module):
    """StratifiedDQE with a CNN patch encoder instead of GraphSAGE.

    The stratum assigner, DQE heads, loss functions, and four-phase training
    protocol are identical to StratifiedDQE.  Only the encoder changes:
    ``encode(patches)`` accepts (B, C, h, w) instead of (N, in_dim) + edge_index.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        embed_dim: int,
        n_strata: int,
        n_intervals: int,
        n_protos: int,
        n_targets: int,
        dropout: float = 0.1,
        K_max: float = 2.0,
        interval_temp: float = 0.5,
        proto_temp: float = 1.0,
    ) -> None:
        super().__init__()
        self.n_strata = n_strata
        self.embed_dim = embed_dim

        self.encoder = PatchCNNEncoder(in_channels, embed_dim, hidden_dim, dropout)
        self.assigner = StratumAssigner(embed_dim, n_strata)
        self.dqes = nn.ModuleList(
            [
                StratumDQE(
                    embed_dim,
                    n_intervals,
                    n_protos,
                    n_targets,
                    K_max,
                    interval_temp,
                    proto_temp,
                )
                for _ in range(n_strata)
            ]
        )
        self.warmup_head = WarmupRegressor(embed_dim, n_targets)

    # ── phase helpers ─────────────────────────────────────────────────────────

    def set_phase(self, phase: int) -> None:
        """Identical freeze/unfreeze logic to StratifiedDQE.set_phase."""
        for p in self.parameters():
            p.requires_grad_(False)
        if phase == 1:
            for p in self.encoder.parameters():
                p.requires_grad_(True)
            for p in self.warmup_head.parameters():
                p.requires_grad_(True)
        elif phase == 2:
            for p in self.assigner.parameters():
                p.requires_grad_(True)
        elif phase == 3:
            for p in self.encoder.parameters():
                p.requires_grad_(True)
            for p in self.assigner.parameters():
                p.requires_grad_(True)
            for dqe in self.dqes:
                for p in dqe.parameters():
                    p.requires_grad_(True)

    # ── forward ───────────────────────────────────────────────────────────────

    def encode(self, patches: torch.Tensor) -> torch.Tensor:  # (B, C, h, w) → (B, d)
        return self.encoder(patches)

    def warmup_forward(self, patches: torch.Tensor) -> torch.Tensor:  # (B, T)
        return self.warmup_head(self.encode(patches))

    def forward(
        self,
        patches: torch.Tensor,  # (B, C, h, w)
        use_wta: bool = False,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Full forward pass — same return signature as StratifiedDQE.forward.

        Returns
        -------
        pred        : (B, T)
        soft_assign : (B, K)
        embeds      : (B, d)
        protos_tan  : (K, n_protos, d)
        kappas      : (K,)
        preds_stack : (B, K, T)
        """
        embeds = self.encode(patches)  # (B, d)
        soft_assign = self.assigner(embeds)  # (B, K)

        preds: list[torch.Tensor] = []
        kappas: list[torch.Tensor] = []
        for dqe in self.dqes:
            assert isinstance(dqe, StratumDQE)
            p, _, _ = dqe(embeds)
            preds.append(p)
            kappas.append(dqe.kappa)

        preds_stack = torch.stack(preds, dim=1)  # (B, K, T)

        if use_wta:
            hard_idx = soft_assign.detach().argmax(dim=-1)
            hard_weights = F.one_hot(hard_idx, self.n_strata).to(soft_assign.dtype)
            routing = hard_weights + soft_assign - soft_assign.detach()
        else:
            routing = soft_assign

        pred = (routing.unsqueeze(-1) * preds_stack).sum(dim=1)  # (B, T)

        kappas_t = torch.stack(kappas)
        protos_tan = torch.stack(
            [dqe.protos_tan for dqe in self.dqes if isinstance(dqe, StratumDQE)]
        )  # (K, P, d)

        return pred, soft_assign, embeds, protos_tan, kappas_t, preds_stack

    # ── k-means initialisation ────────────────────────────────────────────────

    @torch.no_grad()
    def init_from_kmeans(
        self,
        cluster_centres: torch.Tensor,  # (K, d) — CPU preferred
        assign_labels: torch.Tensor,  # (N,)   — CPU preferred
        all_embeds: torch.Tensor,  # (N, d) — CPU preferred
    ) -> None:
        """Device-agnostic k-means initialisation (identical to StratifiedDQE)."""
        self.assigner.init_centroids(cluster_centres)

        labels_cpu = assign_labels.cpu()
        embeds_cpu = all_embeds.cpu()

        for k, dqe in enumerate(self.dqes):
            assert isinstance(dqe, StratumDQE)
            mask = labels_cpu == k
            if mask.sum() == 0:
                continue
            cluster_embeds = embeds_cpu[mask]
            P = dqe.n_protos
            idx = torch.randperm(cluster_embeds.shape[0])[:P]
            if idx.numel() > 0:
                n = idx.numel()
                dqe.protos_tan.data[:n].copy_(cluster_embeds[idx])


# ── Training helpers ──────────────────────────────────────────────────────────


@torch.no_grad()
def collect_embeddings(
    model: StratifiedDQECNN,
    dataset: AirfRANSGridDataset,
    device: torch.device,
    max_embed: int = 200_000,
) -> torch.Tensor:
    """Sample patch embeddings from the training set for k-means.

    Per-graph quota = max_embed // len(dataset).  Patch tensors are moved to
    device one graph at a time and freed immediately after encoding.
    """
    model.eval()
    quota = max(1, max_embed // len(dataset))
    all_embeds: list[torch.Tensor] = []
    for gg in dataset:
        patches_d = gg.patches.to(device)  # (P, C, h, w)
        e = model.encode(patches_d)  # (P, d)
        if e.shape[0] > quota:
            idx = torch.randperm(e.shape[0], device=device)[:quota]
            e = e[idx]
        all_embeds.append(e.cpu())
        del patches_d, e
    return torch.cat(all_embeds, dim=0)  # (total_patches, d)


def kmeans_cluster(
    embeds: torch.Tensor,
    n_clusters: int,
    n_iter: int = 100,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Lloyd k-means in Euclidean space.  Returns (centroids (K, d), labels (N,))."""
    torch.manual_seed(seed)
    N = embeds.shape[0]
    perm = torch.randperm(N)[:n_clusters]
    centroids = embeds[perm].clone()
    labels = torch.zeros(N, dtype=torch.long)

    for _ in range(n_iter):
        dists = torch.cdist(embeds, centroids)  # (N, K)
        new_labels = dists.argmin(dim=-1)
        if (new_labels == labels).all():
            break
        labels = new_labels
        for k in range(n_clusters):
            mask = labels == k
            if mask.sum() > 0:
                centroids[k] = embeds[mask].mean(0)

    return centroids, labels


def reconstruct_node_predictions(
    patch_preds: torch.Tensor,  # (P, T)
    positions: torch.Tensor,  # (P, 2) int — (row, col) patch top-left
    node_grid_rc: torch.Tensor,  # (N, 2) int — (row, col) per node
) -> torch.Tensor:  # (N, T)
    """Reconstruct per-node predictions by averaging overlapping patches.

    Builds a (GRID_H, GRID_W, T) accumulator, adds each patch prediction to all
    covered cells, then looks up each node's cell.  Computationally cheap for
    the default 128×128 grid with 49 patches.
    """
    T = patch_preds.shape[-1]
    dev = patch_preds.device
    acc = torch.zeros(GRID_H, GRID_W, T, device=dev)
    cnt = torch.zeros(GRID_H, GRID_W, device=dev)

    for pi in range(len(positions)):
        ri, ci = int(positions[pi, 0]), int(positions[pi, 1])
        acc[ri : ri + PATCH_H, ci : ci + PATCH_W] += patch_preds[
            pi
        ]  # broadcasts (T,) over grid slice
        cnt[ri : ri + PATCH_H, ci : ci + PATCH_W] += 1.0

    pred_grid = acc / cnt.clamp(min=1).unsqueeze(-1)  # (H, W, T)

    rows = node_grid_rc[:, 0].long()
    cols = node_grid_rc[:, 1].long()
    return pred_grid[rows, cols]  # (N, T)


@torch.no_grad()
def evaluate(
    model: StratifiedDQECNN,
    dataset: AirfRANSGridDataset,
    device: torch.device,
    y_mean: torch.Tensor,
    y_std: torch.Tensor,
    surf_weight: float = 10.0,
) -> dict[str, float]:
    """Streaming per-node evaluation: weighted MSE + per-target R².

    For each graph:
    1. Encode all patches → per-patch predictions.
    2. Reconstruct a (GRID_H, GRID_W, T) prediction grid.
    3. Look up per-node predictions using precomputed grid indices.
    4. Denormalise and accumulate error statistics.
    """
    model.eval()
    y_mean_cpu = y_mean.cpu()
    y_std_cpu = y_std.cpu()

    surf_se = torch.zeros(N_TARGETS)
    vol_se = torch.zeros(N_TARGETS)
    surf_n = 0
    vol_n = 0

    res_sq = torch.zeros(N_TARGETS)
    true_sum = torch.zeros(N_TARGETS)
    true_sq = torch.zeros(N_TARGETS)
    total_n = 0

    for gg in dataset:
        patches_d = gg.patches.to(device)  # (P, C, h, w)
        pred_p, _, _, _, _, _ = model(patches_d)  # (P, T) normalised
        pred_p_cpu = pred_p.cpu()
        del patches_d, pred_p

        pred_n = reconstruct_node_predictions(
            pred_p_cpu, gg.positions, gg.node_grid_rc
        )  # (N, T) normalised

        pred_dn = pred_n * y_std_cpu + y_mean_cpu  # (N, T) denormalised
        true_dn = gg.node_targets * y_std_cpu + y_mean_cpu  # (N, T) denormalised
        smask = gg.node_surf  # (N,)

        se = (pred_dn - true_dn) ** 2
        surf_se += se[smask].sum(0)
        vol_se += se[~smask].sum(0)
        surf_n += int(smask.sum())
        vol_n += int((~smask).sum())

        res_sq += se.sum(0)
        true_sum += true_dn.sum(0)
        true_sq += (true_dn**2).sum(0)
        total_n += true_dn.shape[0]

    T = N_TARGETS
    surf_mse = float(surf_se.sum()) / max(surf_n * T, 1)
    vol_mse = float(vol_se.sum()) / max(vol_n * T, 1)
    mse_w = (surf_mse * surf_weight + vol_mse) / (surf_weight + 1)

    true_mean = true_sum / total_n
    true_var = (true_sq / total_n - true_mean**2).clamp(min=1e-12)
    r2_per = (1.0 - res_sq / (total_n * true_var)).tolist()

    return {
        "weighted_mse": mse_w,
        **{f"r2_{TARGET_NAMES[t]}": float(r2_per[t]) for t in range(N_TARGETS)},
        "r2_mean": float(np.mean(r2_per)),
    }


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

    # ── Load data ─────────────────────────────────────────────────────────────
    # One AirfRANS instance per split; shared across all views to avoid
    # duplicating the in-memory dataset.
    print(f"Loading AirfRANS (task={args.task}) …")
    raw_train = AirfRANS(root=args.data_root, task=args.task, train=True)
    raw_test = AirfRANS(root=args.data_root, task=args.task, train=False)

    n_total = len(raw_train)
    val_size = max(1, int(n_total * 0.1))
    train_indices: list[int] = list(range(n_total - val_size))
    val_indices: list[int] = list(range(n_total - val_size, n_total))

    # ── Normalisation stats ────────────────────────────────────────────────
    print("  Computing normalisation stats (streaming) …")
    norm_ds = AirfRANSGridDataset(
        args.data_root,
        "train",
        task=args.task,
        indices=train_indices,
        _shared_raw=raw_train,
    )
    norm_stats = compute_normalisation_streaming(norm_ds)
    del norm_ds
    gc.collect()
    _, _, y_mean, y_std = norm_stats

    # ── Build datasets ─────────────────────────────────────────────────────
    # Grid caches are independent per dataset object; that is intentional:
    # train/val/test have different norm_stats (test reuses train's) and
    # there is no KNN edge cache to share (no edge building at all).
    train_dataset = AirfRANSGridDataset(
        args.data_root,
        "train",
        task=args.task,
        norm_stats=norm_stats,
        indices=train_indices,
        _shared_raw=raw_train,
    )
    val_dataset = AirfRANSGridDataset(
        args.data_root,
        "train",
        task=args.task,
        norm_stats=norm_stats,
        indices=val_indices,
        _shared_raw=raw_train,
    )
    test_dataset = AirfRANSGridDataset(
        args.data_root,
        "test",
        task=args.task,
        norm_stats=norm_stats,
        _shared_raw=raw_test,
    )

    print(
        f"  {len(train_dataset)} train / {len(val_dataset)} val / "
        f"{len(test_dataset)} test graphs\n"
        f"  Grid: {GRID_H}×{GRID_W}  Patches per graph: {N_PATCHES} "
        f"({PATCH_H}×{PATCH_W}, stride {PATCH_STRIDE})"
    )

    # ── Build model ────────────────────────────────────────────────────────
    model = StratifiedDQECNN(
        in_channels=IN_CHANNELS,
        hidden_dim=args.hidden_dim,
        embed_dim=args.embed_dim,
        n_strata=args.n_strata,
        n_intervals=args.n_intervals,
        n_protos=args.n_protos,
        n_targets=N_TARGETS,
        dropout=args.dropout,
        K_max=args.K_max,
        interval_temp=args.interval_temp,
        proto_temp=args.proto_temp,
    ).to(device)

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        print(f"Resumed from {args.resume}")

    y_mean = y_mean.to(device)
    y_std = y_std.to(device)

    history: dict[str, list[float]] = {
        "total": [],
        "regression": [],
        "entropy": [],
        "diversity": [],
        "centripetal": [],
    }
    kappa_hist: list[list[float]] = []
    phase_boundaries: list[int] = []

    # ── Phase 1: Warmup regressor ──────────────────────────────────────────
    print(f"\n=== Phase 1: Warmup regressor ({args.warmup_epochs} epochs) ===")
    model.set_phase(1)
    opt1 = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=1e-4,
    )
    sched1 = CosineAnnealingLR(opt1, T_max=max(1, args.warmup_epochs))

    for epoch in range(args.warmup_epochs):
        model.train()
        ep_loss = 0.0
        n_graphs = 0
        train_dataset.shuffle()
        for gg in train_dataset:
            patches_d = gg.patches.to(device)  # (P, C, h, w)
            targets_d = gg.patch_targets.to(device)  # (P, T)

            opt1.zero_grad()
            pred = model.warmup_forward(patches_d)  # (P, T)
            loss = F.mse_loss(pred, targets_d)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt1.step()
            ep_loss += loss.item()
            n_graphs += 1
            del patches_d, targets_d, pred

        sched1.step()
        avg = ep_loss / max(1, n_graphs)
        if (epoch + 1) % max(1, args.warmup_epochs // 5) == 0:
            print(f"  [warmup] epoch {epoch + 1:3d}  loss={avg:.4f}")

    phase_boundaries.append(args.warmup_epochs)

    # ── Phase 2: K-means clustering ────────────────────────────────────────
    print("\n=== Phase 2: K-means clustering ===")
    torch.cuda.empty_cache()
    all_embeds = collect_embeddings(
        model, train_dataset, device, args.max_embed_patches
    )
    centres, labels = kmeans_cluster(all_embeds, args.n_strata, seed=args.seed)
    model.init_from_kmeans(centres, labels, all_embeds)
    print(
        f"  Cluster sizes: {[(labels == k).sum().item() for k in range(args.n_strata)]}"
    )

    centres_gpu = centres.to(device)  # (K, d) pseudo-label source

    # ── Phase 3: Stratum assigner pre-train ───────────────────────────────
    print(f"\n=== Phase 3: Stratum assigner ({args.assigner_epochs} epochs) ===")
    model.set_phase(2)
    opt2 = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr * 2,
        weight_decay=1e-4,
    )

    for epoch in range(args.assigner_epochs):
        model.train()
        ep_loss = 0.0
        n_graphs = 0
        train_dataset.shuffle()
        for gg in train_dataset:
            patches_d = gg.patches.to(device)

            opt2.zero_grad()
            with torch.no_grad():
                emb = model.encode(patches_d)  # (P, d)
                diff = emb.unsqueeze(1) - centres_gpu.unsqueeze(0)
                pseudo_labels = (diff**2).sum(-1).argmin(dim=-1)  # (P,)
            soft = model.assigner(emb)
            loss = F.nll_loss(soft.clamp(min=1e-8).log(), pseudo_labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt2.step()
            ep_loss += loss.item()
            n_graphs += 1
            del patches_d, emb, soft

        if (epoch + 1) % max(1, args.assigner_epochs // 5) == 0:
            print(
                f"  [assigner] epoch {epoch + 1:3d}  ce={ep_loss / max(1, n_graphs):.4f}"
            )

    phase_boundaries.append(args.warmup_epochs + args.assigner_epochs)

    # ── Phase 4: Full joint training ───────────────────────────────────────
    print(f"\n=== Phase 4: Full joint training ({args.full_epochs} epochs) ===")
    model.set_phase(3)
    opt3 = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=1e-4,
    )
    sched3 = CosineAnnealingLR(opt3, T_max=max(1, args.full_epochs))

    best_val_mse = math.inf
    best_epoch = 0

    for epoch in range(args.full_epochs):
        model.train()
        ep_break: dict[str, float] = {k: 0.0 for k in history}
        train_dataset.shuffle()

        n_graphs = 0
        for gg in train_dataset:
            patches_d = gg.patches.to(device)  # (P, C, h, w)
            targets_d = gg.patch_targets.to(device)  # (P, T)
            surf_d = gg.patch_surf.to(device)  # (P,)

            opt3.zero_grad()
            use_wta = (
                args.specialization_mode == "wta"
                and random.random() < args.wta_fraction
            )
            (
                pred,
                soft_assign,
                embeds,
                protos_tan,
                kappas,
                preds_stack,
            ) = model(patches_d, use_wta=use_wta)

            protos_mean = protos_tan.mean(dim=1)  # (K, d)

            loss, breakdown = total_loss(
                pred=pred,
                target=targets_d,
                surf_mask=surf_d,
                soft_assign=soft_assign,
                embeds_tan=embeds,
                protos_tan=protos_mean,
                kappas=kappas,
                surf_weight=args.surf_weight,
                entropy_weight=args.entropy_weight,
                diversity_weight=args.diversity_weight,
                centripetal_weight=args.centripetal_weight,
                curvature_margin=args.curvature_margin,
            )

            if args.specialization_mode == "diversity":
                loss = loss + args.specialization_weight * prediction_diversity_loss(
                    preds_stack
                )
            elif args.specialization_mode == "cond_entropy":
                loss = loss + args.specialization_weight * conditional_diversity_loss(
                    preds_stack, soft_assign
                )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt3.step()

            for k, v in breakdown.items():
                ep_break[k] = ep_break.get(k, 0.0) + v
            n_graphs += 1
            del patches_d, targets_d, surf_d, pred, soft_assign, embeds

        sched3.step()

        for k in history:
            history[k].append(ep_break[k] / max(1, n_graphs))

        with torch.no_grad():
            kappas_now = [
                dqe.kappa.item()  # type: ignore[union-attr]
                for dqe in model.dqes
                if isinstance(dqe, StratumDQE)
            ]
        kappa_hist.append(kappas_now)

        if (epoch + 1) % max(1, args.full_epochs // 10) == 0:
            torch.cuda.empty_cache()
            val_metrics = evaluate(
                model, val_dataset, device, y_mean, y_std, surf_weight=args.surf_weight
            )
            print(
                f"  [full] epoch {epoch + 1:3d}  "
                f"total={history['total'][-1]:.4f}  "
                f"val_mse={val_metrics['weighted_mse']:.4f}  "
                f"val_r2={val_metrics['r2_mean']:.3f}  "
                f"kappas={[f'{k:.3f}' for k in kappas_now]}"
            )
            if val_metrics["weighted_mse"] < best_val_mse:
                best_val_mse = val_metrics["weighted_mse"]
                best_epoch = epoch + 1
                torch.save(
                    {
                        "model": model.state_dict(),
                        "epoch": epoch + 1,
                        "val_metrics": val_metrics,
                    },
                    out_dir / "checkpoint.pt",
                )

    print(f"\nBest val MSE = {best_val_mse:.4f} at epoch {best_epoch}")

    # ── Final evaluation ───────────────────────────────────────────────────
    print("\n=== Final evaluation on test set ===")
    ckpt = torch.load(out_dir / "checkpoint.pt", map_location=device)
    model.load_state_dict(ckpt["model"])

    torch.cuda.empty_cache()
    test_metrics = evaluate(
        model, test_dataset, device, y_mean, y_std, surf_weight=args.surf_weight
    )
    print("Test metrics:")
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.4f}")

    with open(out_dir / "test_metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=2)

    # ── Visualisations ─────────────────────────────────────────────────────
    print("\nSaving visualisations …")

    plot_training_curves(
        history, out_dir / "training_curves.png", phase_boundaries=phase_boundaries
    )

    if kappa_hist:
        plot_curvature_evolution(kappa_hist, out_dir / "curvature_evolution.png")

    final_kappas = [
        dqe.kappa.item()  # type: ignore[union-attr]
        for dqe in model.dqes
        if isinstance(dqe, StratumDQE)
    ]
    plot_geometry_summary(final_kappas, out_dir / "geometry_summary.png")

    print(f"All outputs saved to {out_dir}")


# ── CLI ───────────────────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train StratifiedDQE on AirfRANS (CNN patch encoder)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data
    p.add_argument(
        "--data_root", default="../data/AirfRANS", help="AirfRANS download dir"
    )
    p.add_argument(
        "--task",
        default="scarce",
        choices=["scarce", "full"],
        help="'scarce' (~200 train sims) or 'full' (~800 train sims)",
    )

    # Model
    p.add_argument("--n_strata", type=int, default=4, help="Number of strata K")
    p.add_argument("--hidden_dim", type=int, default=128, help="CNN hidden channels")
    p.add_argument("--embed_dim", type=int, default=64, help="Embedding dimension d")
    p.add_argument(
        "--n_intervals", type=int, default=8, help="DQE intervals per stratum"
    )
    p.add_argument("--n_protos", type=int, default=8, help="DQE prototypes per stratum")
    p.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    p.add_argument("--K_max", type=float, default=2.0, help="Max |curvature|")
    p.add_argument(
        "--interval_temp", type=float, default=0.5, help="Interval softmax temperature"
    )
    p.add_argument(
        "--proto_temp", type=float, default=1.0, help="Prototype softmax temperature"
    )

    # Training phases
    p.add_argument("--warmup_epochs", type=int, default=30, help="Phase 1 epochs")
    p.add_argument("--assigner_epochs", type=int, default=10, help="Phase 3 epochs")
    p.add_argument("--full_epochs", type=int, default=100, help="Phase 4 epochs")
    p.add_argument("--lr", type=float, default=3e-4, help="Learning rate")

    # Loss weights
    p.add_argument(
        "--surf_weight", type=float, default=10.0, help="Surface patch loss weight"
    )
    p.add_argument(
        "--entropy_weight", type=float, default=0.1, help="Stratum entropy loss weight"
    )
    p.add_argument(
        "--diversity_weight",
        type=float,
        default=0.05,
        help="Curvature diversity weight",
    )
    p.add_argument(
        "--centripetal_weight", type=float, default=0.01, help="Centripetal loss weight"
    )
    p.add_argument(
        "--curvature_margin", type=float, default=0.1, help="Min curvature separation"
    )

    # Stratum specialisation
    p.add_argument(
        "--specialization_mode",
        default="none",
        choices=["none", "diversity", "wta", "cond_entropy"],
        help=(
            "Stratum specialization mode for Phase 4.  "
            "'cond_entropy' conflicts with --entropy_weight > 0."
        ),
    )
    p.add_argument(
        "--specialization_weight",
        type=float,
        default=0.1,
        help="Weight for specialization loss (diversity / cond_entropy).",
    )
    p.add_argument(
        "--wta_fraction",
        type=float,
        default=0.5,
        help="Fraction of Phase 4 steps using hard WTA routing (mode: wta).",
    )

    # Misc
    p.add_argument("--out_dir", default="results/airfrans_cnn", help="Output directory")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--max_embed_patches",
        type=int,
        default=10_000,
        help=(
            "Max total patches collected for k-means "
            "(budget is split evenly across graphs; each graph has ~49 patches)."
        ),
    )
    p.add_argument(
        "--cpu", action="store_true", help="Force CPU even if CUDA available"
    )
    p.add_argument("--resume", default="", help="Path to checkpoint to resume from")

    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    t0 = time.time()
    train(args)
    print(f"\nTotal wall time: {(time.time() - t0) / 60:.1f} min")
