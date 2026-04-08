"""
visualize_shape_pressure.py — Standalone post-hoc analysis script.

Loads a trained joint pressure+shape DQE checkpoint (produced by run.py) and
regenerates the shape_pressure_surface.png that run.py produces automatically
after training.  Useful for re-running the plot with different colour maps,
zoom levels, or after updating the visualisation code without retraining.

The actual visualisation logic (visualize_shape_pressure, collect_shape_vectors,
compute_shape_vector) lives in run.py — this script only handles checkpoint
loading, dataset construction, and the array collection, then delegates to
those shared functions.

Speed notes
-----------
- Norm stats are read directly from the checkpoint (saved by run.py).
- Shape vectors are computed in parallel with ProcessPoolExecutor.
- All heavy arrays (embeddings + shape matrix) are cached to a .npz file
  (--cache, default next to the checkpoint).  Re-runs skip everything except
  plotting.  Delete the .npz or pass --no_cache to force a rebuild.

Usage
-----
  python visualize_shape_pressure.py \\
      --checkpoint results/pressure_dqe/checkpoint.pt \\
      --data_root ../data/AirfRANS \\
      --out results/pressure_dqe/shape_pressure_surface.png

  # Force cache rebuild:
  python visualize_shape_pressure.py ... --no_cache

  # Match the near_field_y used during training if you changed the default:
  python visualize_shape_pressure.py ... --near_field_y 0.8
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch_geometric.datasets import AirfRANS

from run import (
    NEAR_FIELD_BOUNDS,
    AirfRANSNearFieldDataset,
    GraphGrid,
    PressureDQEModel,
    collect_airfoil_embeddings,
    collect_shape_vectors,
    visualize_shape_pressure,
)


# ── Architecture inference ─────────────────────────────────────────────────────


def infer_model_dims(state_dict: dict) -> dict:
    """Recover PressureDQEModel constructor kwargs from a saved state_dict."""
    sd = state_dict
    in_channels = int(sd["encoder.convs.0.weight"].shape[1])
    hidden_dim  = int(sd["encoder.convs.6.weight"].shape[1])
    embed_dim   = int(sd["encoder.convs.6.weight"].shape[0])
    n_intervals = int(sd["pressure_dqe.interval_center"].shape[0])
    n_protos    = int(sd["pressure_dqe.protos_tan"].shape[0])
    return dict(
        in_channels=in_channels,
        hidden_dim=hidden_dim,
        embed_dim=embed_dim,
        n_intervals=n_intervals,
        n_protos=n_protos,
    )


# ── Cache helpers ──────────────────────────────────────────────────────────────


def _cache_path_from_checkpoint(checkpoint: Path) -> Path:
    return checkpoint.with_suffix(".vis_cache.npz")


def _save_cache(
    path: Path,
    all_embs: np.ndarray,
    all_pvals: np.ndarray,
    all_shape: np.ndarray,
    p_mean: float,
    p_std: float,
) -> None:
    np.savez_compressed(
        path,
        all_embs=all_embs,
        all_pvals=all_pvals,
        all_shape=all_shape,
        p_mean=np.float64(p_mean),
        p_std=np.float64(p_std),
    )
    print(f"  Cache saved → {path}")


def _load_cache(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    d = np.load(path)
    return (
        d["all_embs"],
        d["all_pvals"],
        d["all_shape"],
        float(d["p_mean"]),
        float(d["p_std"]),
    )


# ── Main ──────────────────────────────────────────────────────────────────────


def main(args: argparse.Namespace) -> None:
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    )

    ckpt_path = Path(args.checkpoint)
    cache_path = Path(args.cache) if args.cache else _cache_path_from_checkpoint(ckpt_path)

    # ── Checkpoint ────────────────────────────────────────────────────────
    ckpt = torch.load(ckpt_path, map_location=device)
    dims = infer_model_dims(ckpt["model"])
    print(
        f"Loaded checkpoint  epoch={ckpt['epoch']}  "
        + "  ".join(f"{k}={v}" for k, v in dims.items())
    )

    # Norm stats — read from checkpoint or fall back to a dataset scan
    if "norm_stats" in ckpt:
        ns = ckpt["norm_stats"]
        p_mean_t: torch.Tensor = ns["p_mean"].cpu()
        p_std_t: torch.Tensor = ns["p_std"].cpu()
        print("  Norm stats loaded from checkpoint.")
    else:
        from run import compute_normalisation  # noqa: PLC0415

        NEAR_FIELD_BOUNDS["y_min"] = -args.near_field_y
        NEAR_FIELD_BOUNDS["y_max"] = args.near_field_y
        print("  Computing normalisation stats (not in checkpoint) …")
        raw_train_tmp = AirfRANS(root=args.data_root, task=args.task, train=True)
        n_tmp = len(raw_train_tmp)
        val_tmp = max(1, int(n_tmp * 0.1))
        _, _, p_mean_t, p_std_t = compute_normalisation(
            raw_train_tmp, list(range(n_tmp - val_tmp))
        )
        del raw_train_tmp

    p_mean_v = p_mean_t.item()
    p_std_v = p_std_t.item()

    NEAR_FIELD_BOUNDS["y_min"] = -args.near_field_y
    NEAR_FIELD_BOUNDS["y_max"] = args.near_field_y

    # ── Try cache ─────────────────────────────────────────────────────────
    if cache_path.exists() and not args.no_cache:
        print(f"  Loading arrays from cache {cache_path} …")
        all_embs, all_pvals, all_shape, p_mean_v, p_std_v = _load_cache(cache_path)
    else:
        # ── Raw data ──────────────────────────────────────────────────────
        print(f"Loading AirfRANS (task={args.task}) …")
        raw_train = AirfRANS(root=args.data_root, task=args.task, train=True)
        raw_test  = AirfRANS(root=args.data_root, task=args.task, train=False)

        n_total  = len(raw_train)
        val_size = max(1, int(n_total * 0.1))
        train_idx = list(range(n_total - val_size))
        val_idx   = list(range(n_total - val_size, n_total))

        norm_stats = (
            torch.zeros(1), torch.ones(1),  # x_mean/std not needed for visualisation
            p_mean_t, p_std_t,
        )

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
            norm_stats=norm_stats, _shared_raw=raw_test,
        )
        print(f"  {len(train_ds)} train / {len(val_ds)} val / {len(test_ds)} test")

        # ── Model ─────────────────────────────────────────────────────────
        model = PressureDQEModel(**dims).to(device)
        model.load_state_dict(ckpt["model"])
        model.eval()

        # ── Joint DQE embeddings + mean Cp ────────────────────────────────
        print("  Collecting joint embeddings …")
        with torch.no_grad():
            embs_tr, pv_tr = collect_airfoil_embeddings(model, train_ds, device)
            embs_va, pv_va = collect_airfoil_embeddings(model, val_ds, device)
            embs_te, pv_te = collect_airfoil_embeddings(model, test_ds, device)

        all_embs  = torch.cat([embs_tr, embs_va, embs_te]).numpy().astype(np.float64)
        all_pvals = torch.cat([pv_tr, pv_va, pv_te]).numpy()

        # ── Geometric shape vectors (parallel) ────────────────────────────
        print("  Computing geometric shape vectors …")
        shape_tr = collect_shape_vectors(raw_train, train_ds)
        shape_va = collect_shape_vectors(raw_train, val_ds)
        shape_te = collect_shape_vectors(raw_test,  test_ds)
        all_shape = np.concatenate([shape_tr, shape_va, shape_te])

        _save_cache(cache_path, all_embs, all_pvals, all_shape, p_mean_v, p_std_v)

    # ── Plot ──────────────────────────────────────────────────────────────
    print("  Rendering figure …")
    visualize_shape_pressure(
        all_embs, all_pvals, all_shape,
        p_mean=p_mean_v, p_std=p_std_v,
        out_path=Path(args.out),
    )


# ── CLI ───────────────────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Regenerate the joint pressure+shape similarity surface from a checkpoint",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--checkpoint",
        default="results/pressure_dqe/checkpoint.pt",
        help="Path to checkpoint saved by run.py",
    )
    p.add_argument("--data_root", default="../data/AirfRANS")
    p.add_argument("--task", default="scarce", choices=["scarce", "full"])
    p.add_argument(
        "--near_field_y",
        type=float,
        default=0.6,
        help="Must match the value used during training",
    )
    p.add_argument(
        "--out",
        default="results/pressure_dqe/shape_pressure_surface.png",
        help="Output figure path",
    )
    p.add_argument(
        "--cache",
        default="",
        help="Path for embedding/shape cache .npz (default: checkpoint.vis_cache.npz)",
    )
    p.add_argument(
        "--no_cache",
        action="store_true",
        help="Ignore existing cache and recompute everything",
    )
    p.add_argument("--cpu", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    main(_parse_args())
