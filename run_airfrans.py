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


def load_airfrans(root: str, split: str, knn_k: int = 8) -> list[Data]:
    """Load AirfRANS split, build KNN edges, prepend pos to features.

    Returns a list of Data objects, each with:
      x          (N, 7)  float32
      edge_index (2, E)  long
      y          (N, 4)  float32
      surf       (N,)    bool
      pos        (N, 2)  float32
    """
    transform = KNNGraph(k=knn_k, loop=False, force_undirected=True)
    ds = AirfRANS(root=root, task="full", train=(split == "train"), transform=transform)

    data_list: list[Data] = []
    for sample in ds:
        # AirfRANS: x shape (N, 5), pos shape (N, 2)
        x = sample.x.float() if sample.x is not None else sample.pos.float()
        pos = (
            sample.pos.float() if sample.pos is not None else torch.zeros(x.shape[0], 2)
        )
        # Concatenate spatial position as extra features
        x_aug = torch.cat([pos, x], dim=-1)  # (N, 7)

        y = (
            sample.y.float()
            if sample.y is not None
            else torch.zeros(x.shape[0], N_TARGETS)
        )

        surf = (
            sample.surf.bool()
            if hasattr(sample, "surf") and sample.surf is not None
            else torch.zeros(x_aug.shape[0], dtype=torch.bool)
        )

        edge_index = sample.edge_index.long()

        data_list.append(
            Data(
                x=x_aug,
                edge_index=edge_index,
                y=y,
                surf=surf,
                pos=pos,
                num_nodes=x_aug.shape[0],
            )
        )
    return data_list


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


def compute_normalisation(
    data_list: list[Data],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute per-feature mean/std over training set for x and y."""
    xs = torch.cat([d.x for d in data_list], dim=0)
    ys = torch.cat([d.y for d in data_list], dim=0)
    x_mean, x_std = xs.mean(0), xs.std(0).clamp(min=1e-6)
    y_mean, y_std = ys.mean(0), ys.std(0).clamp(min=1e-6)
    return x_mean, x_std, y_mean, y_std


def normalise(
    data: Data,
    x_mean: torch.Tensor,
    x_std: torch.Tensor,
    y_mean: torch.Tensor,
    y_std: torch.Tensor,
) -> Data:
    data.x = (data.x - x_mean) / x_std
    data.y = (data.y - y_mean) / y_std
    return data


# ── Training helpers ──────────────────────────────────────────────────────────


@torch.no_grad()
def collect_embeddings(
    model: StratifiedDQE,
    data_list: list[Data],
    device: torch.device,
) -> torch.Tensor:
    """Run encoder over all training samples; return concatenated embeds.

    Always uses the full graph — k-means needs embeddings for every node,
    and no_grad inference is memory-efficient enough on full AirfRANS graphs.
    """
    model.eval()
    all_embeds = []
    for data in data_list:
        d = data.to(device)
        assert d.x is not None and d.edge_index is not None
        e = model.encode(d.x, d.edge_index)
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
    N, d = embeds.shape
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


def evaluate(
    model: StratifiedDQE,
    data_list: list[Data],
    device: torch.device,
    y_mean: torch.Tensor,
    y_std: torch.Tensor,
    surf_weight: float = 10.0,
) -> dict[str, float]:
    """Compute weighted MSE and per-target R² on a data list (full graphs)."""
    model.eval()
    all_pred, all_true, all_surf = [], [], []
    with torch.no_grad():
        for data in data_list:
            d = data.to(device)
            assert d.x is not None and d.edge_index is not None
            y_t = cast(torch.Tensor, d.y)
            surf_t = cast(torch.Tensor, d.surf)
            pred, _, _, _, _, _ = model(d.x, d.edge_index)
            all_pred.append(pred.cpu())
            all_true.append(y_t.cpu())
            all_surf.append(surf_t.cpu())

    pred = torch.cat(all_pred, dim=0)
    true = torch.cat(all_true, dim=0)
    surf = torch.cat(all_surf, dim=0)

    # Denormalise
    pred_dn = pred * y_std + y_mean
    true_dn = true * y_std + y_mean

    mse_w = float(
        F.mse_loss(pred_dn[surf], true_dn[surf]) * surf_weight
        + F.mse_loss(pred_dn[~surf], true_dn[~surf])
    ) / (surf_weight + 1)

    r2_per = []
    for t in range(N_TARGETS):
        var_res = float(((pred_dn[:, t] - true_dn[:, t]) ** 2).mean())
        var_tot = float(true_dn[:, t].var().clamp(min=1e-12))
        r2_per.append(1.0 - var_res / var_tot)

    return {
        "weighted_mse": mse_w,
        **{f"r2_{TARGET_NAMES[t]}": r2_per[t] for t in range(N_TARGETS)},
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
    print("Loading AirfRANS …")
    train_list = load_airfrans(args.data_root, "train", knn_k=args.knn_k)
    test_list = load_airfrans(args.data_root, "test", knn_k=args.knn_k)

    x_mean, x_std, y_mean, y_std = compute_normalisation(train_list)
    train_list = [normalise(d, x_mean, x_std, y_mean, y_std) for d in train_list]
    test_list = [normalise(d, x_mean, x_std, y_mean, y_std) for d in test_list]

    assert train_list[0].x is not None
    in_dim = train_list[0].x.shape[1]  # should be 7
    print(f"  Train graphs: {len(train_list)}, in_dim={in_dim}")

    # Optional validation split (last 10% of train)
    val_size = max(1, int(len(train_list) * 0.1))
    val_list = train_list[-val_size:]
    train_list = train_list[:-val_size]
    print(
        f"  Using {len(train_list)} train / {len(val_list)} val / {len(test_list)} test"
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
        random.shuffle(train_list)
        for data in train_list:
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
    all_embeds = collect_embeddings(model, train_list, device)
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
        random.shuffle(train_list)
        for data in train_list:
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
        random.shuffle(train_list)

        n_batches = 0
        for data in train_list:
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
                float(dqe.kappa.item())
                for dqe in model.dqes  # type: ignore[union-attr]
                if hasattr(dqe, "kappa")
            ]
        kappa_hist.append(kappas_now)

        if (epoch + 1) % max(1, args.full_epochs // 10) == 0:
            val_metrics = evaluate(
                model, val_list, device, y_mean, y_std, surf_weight=args.surf_weight
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
        model, test_list, device, y_mean, y_std, surf_weight=args.surf_weight
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
    sample = test_list[0].to(device)
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
        sample.y,
        out_dir / "prediction_scatter.png",
        surf_mask=sample.surf,
    )

    final_kappas = [
        float(dqe.kappa.item())
        for dqe in model.dqes  # type: ignore[union-attr]
        if hasattr(dqe, "kappa")
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
        "--cpu", action="store_true", help="Force CPU even if CUDA available"
    )
    p.add_argument("--resume", default="", help="Path to checkpoint to resume from")

    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    t0 = time.time()
    train(args)
    print(f"\nTotal wall time: {(time.time() - t0) / 60:.1f} min")
