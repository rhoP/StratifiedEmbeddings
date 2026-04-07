"""
run_airfrans.py — Train & evaluate StratifiedDQE on the AirfRANS dataset.

Usage examples
--------------
  # Full run with defaults
  python run_airfrans.py --out_dir results/airfrans

  # Quicker smoke-test (2 strata, small model, 10 warmup epochs)
  python run_airfrans.py --n_strata 2 --embed_dim 16 --warmup_epochs 10 --full_epochs 20

  # Resume from a saved checkpoint
  python run_airfrans.py --resume results/airfrans/checkpoint.pt

Training phases
---------------
1. Warmup regressor   (--warmup_epochs)
2. K-means cluster    (runs once after warmup, no epochs)
3. Stratum assigner   (--assigner_epochs)
4. Full joint train   (--full_epochs)

AirfRANS specifics
------------------
- Raw dataset has no edge_index; we build a KNN graph (k=--knn_k).
- Node features: 5 (x, y, SDF, normals_x, normals_y)  + 2 (spatial coords)
  → concatenated: in_dim = 7
- Node targets: 4 (u_x, u_y, pressure, ν_t)
- data.surf: boolean mask for airfoil surface nodes (used in loss weighting)
- Each sample is a full CFD simulation (~180 k nodes).  Training uses
  NeighborLoader with --batch_size seed nodes and --num_neighbors fanout
  per hop; inference always runs on the full graph.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import cast

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.datasets import AirfRANS
from torch_geometric.transforms import KNNGraph
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader

from StratifiedEmbedding import (
    StratifiedDQE,
    StratumDQE,
    total_loss,
    prediction_diversity_loss,
    conditional_diversity_loss,
    plot_training_curves,
    plot_curvature_evolution,
    plot_stratum_assignments,
    plot_prediction_scatter,
    plot_geometry_summary,
)

# ── Constants ────────────────────────────────────────────────────────────────

TARGET_NAMES = ["u_x", "u_y", "p", "nu_t"]
N_TARGETS = 4
BASE_IN_DIM = 5  # raw AirfRANS node features (excl. spatial pos)


# ── Data utilities ───────────────────────────────────────────────────────────


class AirfRANSDataset:
    """Lazy per-sample loader backed by PyG's own processed cache.

    Uses ``pre_transform=KNNGraph`` so edges are built once and stored in
    ``<root>/airfrans/processed/``.  No secondary cache is created — disk
    usage equals the PyG processed folder only.

    Feature augmentation (prepend spatial pos to x) and normalisation are
    applied on-the-fly at load time, so the stored tensors remain in their
    original scale.

    Parameters
    ----------
    task : "scarce" | "full"
        AirfRANS task split.  "scarce" (~200 simulations) is the default;
        "full" (~800 training simulations) is available via --task full.
    """

    def __init__(
        self,
        root: str,
        split: str,
        task: str = "scarce",
        knn_k: int = 8,
        norm_stats: tuple[torch.Tensor, torch.Tensor,
                          torch.Tensor, torch.Tensor] | None = None,
        indices: list[int] | None = None,
    ) -> None:
        pre_transform = KNNGraph(k=knn_k, loop=False, force_undirected=True)
        self._raw = AirfRANS(
            root=root,
            task=task,
            train=(split == "train"),
            pre_transform=pre_transform,
        )
        n = len(self._raw)
        self._indices: list[int] = (
            list(indices) if indices is not None else list(range(n))
        )
        self._order: list[int] = list(range(len(self._indices)))
        self.norm_stats = norm_stats

    # ── public API ─────────────────────────────────────────────────────────────

    @property
    def in_dim(self) -> int:
        """Feature dimension — peeked from first graph (always pos + x = 7)."""
        x = self[0].x
        assert x is not None
        return int(x.shape[1])

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, pos: int) -> Data:
        raw: Data = self._raw[self._indices[self._order[pos]]]  # type: ignore[assignment]
        # AirfRANS always supplies x (5 features), pos (2 coords), y (4 targets),
        # surf (bool mask), and edge_index (added by pre_transform).
        x_feat = cast(torch.Tensor, raw.x).float()
        pos_   = cast(torch.Tensor, raw.pos).float()
        x_aug  = torch.cat([pos_, x_feat], dim=-1)   # (N, 7)
        y      = cast(torch.Tensor, raw.y).float()
        surf   = (
            cast(torch.Tensor, raw.surf).bool()
            if hasattr(raw, "surf") and raw.surf is not None
            else torch.zeros(x_aug.shape[0], dtype=torch.bool)
        )
        if self.norm_stats is not None:
            x_mean, x_std, y_mean, y_std = self.norm_stats
            x_aug = (x_aug - x_mean) / x_std
            y     = (y     - y_mean) / y_std
        return Data(
            x=x_aug,
            edge_index=cast(torch.Tensor, raw.edge_index).long(),
            y=y,
            surf=surf,
            num_nodes=x_aug.shape[0],
        )

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def shuffle(self) -> None:
        """Randomly permute the iteration order for the next epoch."""
        random.shuffle(self._order)


def make_neighbor_loader(
    data: Data,
    batch_size: int,
    num_neighbors: list[int],
    shuffle: bool = True,
) -> NeighborLoader:
    """NeighborLoader for GraphSAGE-style mini-batch training.

    Each mini-batch contains `batch_size` seed nodes plus their sampled
    k-hop neighborhoods.  The first `batch.batch_size` rows of every batch
    tensor correspond to the seed nodes; only those rows contribute to the
    loss.  Neighbor rows are used solely for message passing.

    This preserves graph connectivity — every seed node has real neighbors —
    unlike random node subsampling which retains ~(batch_size/N)² of edges.
    """
    return NeighborLoader(
        data,
        num_neighbors=num_neighbors,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
    )


# ── Normalisation ─────────────────────────────────────────────────────────────


def compute_normalisation_streaming(
    dataset: AirfRANSDataset,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Streaming mean/std over all graphs — never holds more than one graph in RAM.

    Uses the online accumulation formula:
        mean  = Σx / N
        var   = Σx² / N − mean²   (cancelled by clamp to handle fp rounding)
    """
    # Peek at first graph to learn feature dimensions, then initialise accumulators.
    first = dataset[0]
    assert first.x is not None
    x0 = first.x.float()
    y0 = cast(torch.Tensor, first.y).float()
    x_sum  = torch.zeros(x0.shape[1])
    x_sum2 = torch.zeros(x0.shape[1])
    y_sum  = torch.zeros(y0.shape[1])
    y_sum2 = torch.zeros(y0.shape[1])
    n_x = n_y = 0

    for data in dataset:
        assert data.x is not None
        x = data.x.float()                        # (N, Fx)
        y = cast(torch.Tensor, data.y).float()    # (N, Fy)
        x_sum  += x.sum(0);  x_sum2 += (x ** 2).sum(0)
        y_sum  += y.sum(0);  y_sum2 += (y ** 2).sum(0)
        n_x += x.shape[0]
        n_y += y.shape[0]

    x_mean = x_sum / n_x
    x_std  = ((x_sum2 / n_x) - x_mean ** 2).clamp(min=0).sqrt().clamp(min=1e-6)
    y_mean = y_sum / n_y
    y_std  = ((y_sum2 / n_y) - y_mean ** 2).clamp(min=0).sqrt().clamp(min=1e-6)
    return x_mean, x_std, y_mean, y_std


# ── Training helpers ──────────────────────────────────────────────────────────


@torch.no_grad()
def collect_embeddings(
    model: StratifiedDQE,
    dataset: AirfRANSDataset,
    device: torch.device,
    max_embed_nodes: int = 500_000,
) -> torch.Tensor:
    """Sample embeddings from each graph for k-means — never all at once.

    Per-graph quota = max_embed_nodes // len(dataset), so total samples stay
    within budget regardless of dataset size.  Full-graph inference runs under
    no_grad; only a random subset is kept in CPU RAM.
    """
    model.eval()
    quota = max(1, max_embed_nodes // len(dataset))
    all_embeds: list[torch.Tensor] = []
    for data in dataset:
        d = data.to(device)
        assert d.x is not None and d.edge_index is not None
        e = model.encode(d.x, d.edge_index)   # (N, d)  — on device
        n = e.shape[0]
        if n > quota:
            idx = torch.randperm(n, device=device)[:quota]
            e = e[idx]
        all_embeds.append(e.cpu())
    return torch.cat(all_embeds, dim=0)


def kmeans_cluster(
    embeds: torch.Tensor,
    n_clusters: int,
    n_iter: int = 100,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Simple Lloyd k-means in Euclidean space.

    Returns (centroids (K, d), labels (N,)).
    """
    torch.manual_seed(seed)
    N = embeds.shape[0]
    perm = torch.randperm(N)[:n_clusters]
    centroids = embeds[perm].clone()

    labels = torch.zeros(N, dtype=torch.long)
    for _ in range(n_iter):
        diff = embeds.unsqueeze(1) - centroids.unsqueeze(0)  # (N, K, d)
        dists = (diff**2).sum(-1)  # (N, K)
        new_labels = dists.argmin(dim=-1)  # (N,)

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
    model: StratifiedDQE,
    dataset: AirfRANSDataset,
    device: torch.device,
    y_mean: torch.Tensor,
    y_std: torch.Tensor,
    surf_weight: float = 10.0,
) -> dict[str, float]:
    """Streaming weighted MSE and per-target R² — never holds all predictions in RAM.

    Accumulates per-graph squared errors and sufficient statistics for R²
    (true_sum, true_sq) so the final metrics are computed from running sums.
    """
    model.eval()
    y_mean_cpu = y_mean.cpu()
    y_std_cpu  = y_std.cpu()

    surf_se = torch.zeros(N_TARGETS)   # Σ squared errors on surface nodes (per target)
    vol_se  = torch.zeros(N_TARGETS)   # Σ squared errors on volume  nodes
    surf_n  = 0
    vol_n   = 0

    res_sq   = torch.zeros(N_TARGETS)  # Σ (pred − true)² per target
    true_sum = torch.zeros(N_TARGETS)  # Σ true, for computing global mean
    true_sq  = torch.zeros(N_TARGETS)  # Σ true², for computing Var[true]
    total_n  = 0

    for data in dataset:
        d = data.to(device)
        assert d.x is not None and d.edge_index is not None
        y_t    = cast(torch.Tensor, d.y)
        surf_t = cast(torch.Tensor, d.surf)
        pred_d, _, _, _, _, _ = model(d.x, d.edge_index)
        pred_dn = (pred_d * y_std + y_mean).cpu()          # (N, T) denormalised
        true_dn = y_t.cpu() * y_std_cpu + y_mean_cpu       # (N, T)
        smask   = surf_t.cpu().bool()                       # (N,)

        se = (pred_dn - true_dn) ** 2                      # (N, T)
        surf_se += se[smask].sum(0)
        vol_se  += se[~smask].sum(0)
        surf_n  += int(smask.sum())
        vol_n   += int((~smask).sum())

        res_sq   += se.sum(0)
        true_sum += true_dn.sum(0)
        true_sq  += (true_dn ** 2).sum(0)
        total_n  += true_dn.shape[0]

    # Weighted MSE: average per-element error, weighted by node type
    T        = N_TARGETS
    surf_mse = float(surf_se.sum()) / max(surf_n * T, 1)
    vol_mse  = float(vol_se.sum())  / max(vol_n  * T, 1)
    mse_w    = (surf_mse * surf_weight + vol_mse) / (surf_weight + 1)

    # R² = 1 − Σ(pred−true)² / (N · Var[true])
    true_mean = true_sum / total_n
    true_var  = (true_sq / total_n - true_mean ** 2).clamp(min=1e-12)
    r2_per    = (1.0 - res_sq / (total_n * true_var)).tolist()

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

    # ── Load data (lazy — one graph at a time, never all in RAM) ─────────────
    print(f"Loading AirfRANS (task={args.task}) …")
    train_raw = AirfRANSDataset(
        args.data_root, "train", task=args.task, knn_k=args.knn_k
    )
    n_total   = len(train_raw)
    val_size  = max(1, int(n_total * 0.1))
    train_indices: list[int] = list(range(n_total - val_size))
    val_indices:   list[int] = list(range(n_total - val_size, n_total))

    print("  Computing normalisation stats (streaming) …")
    norm_stats = compute_normalisation_streaming(
        AirfRANSDataset(
            args.data_root, "train", task=args.task,
            knn_k=args.knn_k, indices=train_indices,
        )
    )
    _, _, y_mean, y_std = norm_stats

    train_dataset = AirfRANSDataset(
        args.data_root, "train", task=args.task,
        knn_k=args.knn_k, norm_stats=norm_stats, indices=train_indices,
    )
    val_dataset = AirfRANSDataset(
        args.data_root, "train", task=args.task,
        knn_k=args.knn_k, norm_stats=norm_stats, indices=val_indices,
    )
    test_dataset = AirfRANSDataset(
        args.data_root, "test", task=args.task,
        knn_k=args.knn_k, norm_stats=norm_stats,
    )

    in_dim = train_dataset.in_dim
    print(
        f"  {len(train_dataset)} train / {len(val_dataset)} val / "
        f"{len(test_dataset)} test graphs, in_dim={in_dim}"
    )

    # ── Build model ────────────────────────────────────────────────────────────
    model = StratifiedDQE(
        in_dim=in_dim,
        hidden_dim=args.hidden_dim,
        embed_dim=args.embed_dim,
        n_strata=args.n_strata,
        n_intervals=args.n_intervals,
        n_protos=args.n_protos,
        n_targets=N_TARGETS,
        n_layers=args.n_layers,
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

    # History dicts
    history: dict[str, list[float]] = {
        "total": [],
        "regression": [],
        "entropy": [],
        "diversity": [],
        "centripetal": [],
    }
    kappa_hist: list[list[float]] = []
    phase_boundaries: list[int] = []

    # ── Phase 1: Warmup regressor ──────────────────────────────────────────────
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
        n_batches = 0
        train_dataset.shuffle()
        for data in train_dataset:
            for batch in make_neighbor_loader(
                data, args.batch_size, args.num_neighbors
            ):
                batch = batch.to(device)
                n_seed = batch.batch_size
                opt1.zero_grad()
                pred_all = model.warmup_forward(batch.x, batch.edge_index)
                loss = F.mse_loss(pred_all[:n_seed], batch.y[:n_seed])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt1.step()
                ep_loss += loss.item()
                n_batches += 1
        sched1.step()
        avg = ep_loss / max(1, n_batches)
        if (epoch + 1) % max(1, args.warmup_epochs // 5) == 0:
            print(f"  [warmup] epoch {epoch + 1:3d}  loss={avg:.4f}")

    phase_boundaries.append(args.warmup_epochs)

    # ── Phase 2: K-means initialisation ───────────────────────────────────────
    print("\n=== Phase 2: K-means clustering ===")
    all_embeds = collect_embeddings(model, train_dataset, device, args.max_embed_nodes)
    centres, labels = kmeans_cluster(all_embeds, args.n_strata, seed=args.seed)
    model.init_from_kmeans(centres.to(device), labels, all_embeds.to(device))
    print(
        f"  Cluster sizes: {[(labels == k).sum().item() for k in range(args.n_strata)]}"
    )

    # Centroids on device — used as pseudo-label source throughout Phase 3.
    centres_gpu = centres.to(device)  # (K, d)

    # ── Phase 3: Stratum assigner pre-train ───────────────────────────────────
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
        n_batches = 0
        train_dataset.shuffle()
        for data in train_dataset:
            for batch in make_neighbor_loader(
                data, args.batch_size, args.num_neighbors
            ):
                batch = batch.to(device)
                n_seed = batch.batch_size
                opt2.zero_grad()
                with torch.no_grad():
                    emb_all = model.encode(batch.x, batch.edge_index)
                    emb = emb_all[:n_seed]
                    # Derive pseudo-labels from k-means centroids for seed nodes.
                    diff = emb.unsqueeze(1) - centres_gpu.unsqueeze(0)  # (N, K, d)
                    pseudo_labels = (diff**2).sum(-1).argmin(dim=-1)  # (N,)
                soft = model.assigner(emb)
                loss = F.nll_loss(soft.clamp(min=1e-8).log(), pseudo_labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt2.step()
                ep_loss += loss.item()
                n_batches += 1
        if (epoch + 1) % max(1, args.assigner_epochs // 5) == 0:
            print(
                f"  [assigner] epoch {epoch + 1:3d}  ce={ep_loss / max(1, n_batches):.4f}"
            )

    phase_boundaries.append(args.warmup_epochs + args.assigner_epochs)

    # ── Phase 4: Full joint training ───────────────────────────────────────────
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

        n_batches = 0
        for data in train_dataset:
            for batch in make_neighbor_loader(
                data, args.batch_size, args.num_neighbors
            ):
                batch = batch.to(device)
                n_seed = batch.batch_size
                opt3.zero_grad()

                # WTA routing: random subset of steps when mode='wta'
                use_wta_step = (
                    args.specialization_mode == "wta"
                    and random.random() < args.wta_fraction
                )
                (
                    pred_all,
                    soft_assign_all,
                    embeds_all,
                    protos_tan,
                    kappas,
                    preds_stack_all,
                ) = model(batch.x, batch.edge_index, use_wta=use_wta_step)
                # Slice to seed nodes — neighbor nodes provide message-passing
                # context only and must not contribute to the loss.
                pred = pred_all[:n_seed]
                soft_assign = soft_assign_all[:n_seed]
                embeds = embeds_all[:n_seed]
                preds_stack = preds_stack_all[:n_seed]  # (n_seed, K, T)

                # Aggregate protos to (K, d) for centripetal_loss.
                protos_mean = protos_tan.mean(dim=1)  # (K, d)

                loss, breakdown = total_loss(
                    pred=pred,
                    target=batch.y[:n_seed],
                    surf_mask=batch.surf[:n_seed],
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

                # Optional specialization loss (options 1 and 3; option 2 is
                # the WTA routing already applied above).
                if args.specialization_mode == "diversity":
                    loss = (
                        loss
                        + args.specialization_weight
                        * prediction_diversity_loss(preds_stack)
                    )
                elif args.specialization_mode == "cond_entropy":
                    loss = (
                        loss
                        + args.specialization_weight
                        * conditional_diversity_loss(preds_stack, soft_assign)
                    )

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt3.step()

                for k, v in breakdown.items():
                    ep_break[k] = ep_break.get(k, 0.0) + v
                n_batches += 1

        sched3.step()

        for k in history:
            history[k].append(ep_break[k] / max(1, n_batches))

        with torch.no_grad():
            kappas_now = [
                dqe.kappa.item()  # type: ignore[union-attr]
                for dqe in model.dqes
                if isinstance(dqe, StratumDQE)
            ]
        kappa_hist.append(kappas_now)

        if (epoch + 1) % max(1, args.full_epochs // 10) == 0:
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

    # ── Final evaluation ───────────────────────────────────────────────────────
    print("\n=== Final evaluation on test set ===")
    ckpt = torch.load(out_dir / "checkpoint.pt", map_location=device)
    model.load_state_dict(ckpt["model"])

    test_metrics = evaluate(
        model, test_dataset, device, y_mean, y_std, surf_weight=args.surf_weight
    )
    print("Test metrics:")
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.4f}")

    with open(out_dir / "test_metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=2)

    # ── Visualisations ─────────────────────────────────────────────────────────
    print("\nSaving visualisations …")

    plot_training_curves(
        history,
        out_dir / "training_curves.png",
        phase_boundaries=phase_boundaries,
    )

    if kappa_hist:
        plot_curvature_evolution(kappa_hist, out_dir / "curvature_evolution.png")

    # Stratum assignment plot (first test sample, no subsampling)
    sample = test_dataset[0].to(device)
    assert sample.x is not None and sample.edge_index is not None
    model.eval()
    with torch.no_grad():
        embeds_vis = model.encode(sample.x, sample.edge_index)
        asgn_vis = model.assigner.hard_assignments(embeds_vis)
        pred_vis, _, _, _, _, _ = model(sample.x, sample.edge_index)

    plot_stratum_assignments(
        embeds_vis,
        asgn_vis,
        out_dir / "stratum_assignments.png",
        surf_mask=sample.surf,
    )

    plot_prediction_scatter(
        pred_vis,
        cast(torch.Tensor, sample.y),
        out_dir / "prediction_scatter.png",
        surf_mask=sample.surf,
    )

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
        description="Train StratifiedDQE on AirfRANS",
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
        help=(
            "AirfRANS task variant.  'scarce' (~200 train sims, fits easily in memory) "
            "is the default.  'full' (~800 train sims) needs more disk and RAM."
        ),
    )
    p.add_argument("--knn_k", type=int, default=8, help="KNN graph edges per node")
    p.add_argument(
        "--batch_size",
        type=int,
        default=2048,
        help="Seed nodes per NeighborLoader mini-batch during training",
    )
    p.add_argument(
        "--num_neighbors",
        type=lambda s: [int(x) for x in s.split(",")],
        default=[25, 10, 10, 10],
        help="Neighbor fanout per GNN hop, comma-separated (e.g. '25,10,10,10'). "
        "Length should match --n_layers.",
    )

    # Model
    p.add_argument("--n_strata", type=int, default=4, help="Number of strata K")
    p.add_argument("--hidden_dim", type=int, default=128, help="GNN hidden dimension")
    p.add_argument("--embed_dim", type=int, default=64, help="Embedding dimension d")
    p.add_argument(
        "--n_intervals", type=int, default=8, help="DQE intervals per stratum"
    )
    p.add_argument("--n_protos", type=int, default=8, help="DQE prototypes per stratum")
    p.add_argument("--n_layers", type=int, default=4, help="GNN layers")
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
        "--surf_weight", type=float, default=10.0, help="Surface node loss weight"
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

    # Stratum specialization (experimental — off by default)
    p.add_argument(
        "--specialization_mode",
        default="none",
        choices=["none", "diversity", "wta", "cond_entropy"],
        help=(
            "Stratum specialization mechanism for Phase 4. "
            "'diversity': adds prediction_diversity_loss (option 1). "
            "'wta': straight-through hard routing on --wta_fraction of steps (option 2). "
            "'cond_entropy': concentrated-assignment + diverse-prediction loss (option 3). "
            "Note: 'cond_entropy' conflicts with --entropy_weight > 0 (set it to 0 for that mode)."
        ),
    )
    p.add_argument(
        "--specialization_weight",
        type=float,
        default=0.1,
        help="Weight for specialization loss (modes: diversity, cond_entropy).",
    )
    p.add_argument(
        "--wta_fraction",
        type=float,
        default=0.5,
        help="Fraction of Phase 4 steps using hard WTA routing (mode: wta).",
    )

    # Misc
    p.add_argument("--out_dir", default="results/airfrans", help="Output directory")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--max_embed_nodes",
        type=int,
        default=500_000,
        help=(
            "Max total nodes sampled from the training set for k-means "
            "(budget is distributed evenly across graphs, so RAM stays bounded)."
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
