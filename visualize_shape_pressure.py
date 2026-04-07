"""
visualize_shape_pressure.py — Load a trained pressure DQE, compute a chord-wise
airfoil shape metric, and superimpose it on the 3-D pressure-similarity surface.

Shape metric
------------
For each airfoil, the raw AirfRANS surface nodes (raw.surf==True) are split into
upper and lower surfaces by y-sign relative to the chord centroid.  Each surface
is interpolated to a shared N_CHORD-point x-grid and stacked into a
2×N_CHORD-dim shape vector [thickness(x), camber(x)].  PCA on the shape matrix
gives continuous shape coordinates that capture thickness and camber variation
across the dataset.

Visualisation (2-panel figure)
-------------------------------
Left  (3-D) — The pressure-interval PCA landscape from run.py training.
              Surface mesh is coloured by mean surface Cp.
              Scatter points are coloured by shape PC1, revealing whether
              shape-similar airfoils cluster together in pressure space.

Right (2-D) — Shape similarity space: each airfoil is a point in shape-PCA
              coordinates, coloured by mean surface Cp.  Comparing left and
              right shows how well the pressure DQE's latent space tracks
              airfoil geometry.

The Pearson correlation between pressure PC1 and shape PC1 is annotated on
both panels.  A high |r| means the model implicitly learned shape.

Usage
-----
  python visualize_shape_pressure.py \\
      --checkpoint results/pressure_dqe/checkpoint.pt \\
      --data_root ../data/AirfRANS \\
      --out results/pressure_dqe/shape_pressure_surface.png

  # Match the near_field_y used during training if you changed the default:
  python visualize_shape_pressure.py ... --near_field_y 0.8
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import cast

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import torch
from mpl_toolkits.mplot3d import (
    Axes3D,
)  # noqa: F401 — registers 3-D projection in mpl < 3.4
from scipy.interpolate import griddata
from scipy.stats import pearsonr
from torch_geometric.data import Data
from torch_geometric.datasets import AirfRANS

from run import (
    IN_CHANNELS,
    NEAR_FIELD_BOUNDS,
    AirfRANSNearFieldDataset,
    GraphGrid,
    PressureDQEModel,
    collect_airfoil_embeddings,
    compute_normalisation,
)

# ── Shape metric constants ─────────────────────────────────────────────────────

N_CHORD: int = 128  # chord-wise interpolation points for thickness/camber vectors


# ── Shape metric ──────────────────────────────────────────────────────────────


def compute_shape_vector(surf_pos: np.ndarray) -> np.ndarray:
    """Compute a [thickness(x), camber(x)] vector from airfoil surface nodes.

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

    # Normalise chord to [0, 1] — AirfRANS airfoils have unit chord but
    # compute it explicitly to be safe.
    x_min, x_max = x.min(), x.max()
    chord = x_max - x_min
    if chord < 1e-6:
        return np.zeros(2 * N_CHORD)
    x_n = (x - x_min) / chord

    # Split upper/lower by y relative to the chord-midline.
    y_mid = y.mean()
    upper = y >= y_mid
    lower = ~upper

    if upper.sum() < 6 or lower.sum() < 6:
        return np.zeros(2 * N_CHORD)

    x_grid = np.linspace(0.0, 1.0, N_CHORD)

    su = np.argsort(x_n[upper])
    y_up = np.interp(x_grid, x_n[upper][su], y[upper][su])

    sl = np.argsort(x_n[lower])
    y_lo = np.interp(x_grid, x_n[lower][sl], y[lower][sl])

    thickness = y_up - y_lo  # ≥ 0 for a physically valid airfoil
    camber = (y_up + y_lo) / 2.0  # signed camber above/below chord

    return np.concatenate([thickness, camber])  # (2 * N_CHORD,)


def collect_shape_vectors(
    raw_ds: AirfRANS,
    dataset: AirfRANSNearFieldDataset,
) -> np.ndarray:
    """Collect shape vectors in the same iteration order as collect_airfoil_embeddings.

    Iterates the dataset via its current _order list (identical sequential order
    used by collect_airfoil_embeddings when no shuffle has been called), so the
    i-th row corresponds to the i-th embedding returned by that function.

    Returns
    -------
    (A, 2 * N_CHORD) shape matrix
    """
    vecs: list[np.ndarray] = []
    for i in range(len(dataset)):
        raw_idx = dataset._indices[dataset._order[i]]
        raw: Data = raw_ds[raw_idx]  # type: ignore[assignment]

        pos = cast(torch.Tensor, raw.pos).float().numpy()
        surf_mask = (
            cast(torch.Tensor, raw.surf).bool().numpy()
            if hasattr(raw, "surf") and raw.surf is not None
            else np.zeros(pos.shape[0], dtype=bool)
        )
        vecs.append(compute_shape_vector(pos[surf_mask]))

    return np.stack(vecs)  # (A, 2 * N_CHORD)


# ── Architecture inference ─────────────────────────────────────────────────────


def infer_model_dims(state_dict: dict) -> dict:
    """Recover PressureDQEModel constructor kwargs from a saved state_dict.

    Reads tensor shapes from known parameter keys so no separate hparam file
    is needed.  Keys relied on:
      encoder.convs.0.weight  — (hidden//4, in_channels, 3, 3)
      encoder.convs.6.weight  — (embed_dim, hidden_dim, 3, 3)
      dqe.interval_center     — (n_intervals,)
      dqe.protos_tan          — (n_protos, embed_dim)
    """
    sd = state_dict
    in_channels = int(sd["encoder.convs.0.weight"].shape[1])
    hidden_dim = int(sd["encoder.convs.6.weight"].shape[1])
    embed_dim = int(sd["encoder.convs.6.weight"].shape[0])
    n_intervals = int(sd["dqe.interval_center"].shape[0])
    n_protos = int(sd["dqe.protos_tan"].shape[0])
    return dict(
        in_channels=in_channels,
        hidden_dim=hidden_dim,
        embed_dim=embed_dim,
        n_intervals=n_intervals,
        n_protos=n_protos,
    )


# ── Main ──────────────────────────────────────────────────────────────────────


def main(args: argparse.Namespace) -> None:
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    )

    # ── Checkpoint ────────────────────────────────────────────────────────
    ckpt = torch.load(args.checkpoint, map_location=device)
    dims = infer_model_dims(ckpt["model"])
    print(
        f"Loaded checkpoint  epoch={ckpt['epoch']}  "
        + "  ".join(f"{k}={v}" for k, v in dims.items())
    )

    # Apply near-field domain bounds before any dataset construction.
    NEAR_FIELD_BOUNDS["y_min"] = -args.near_field_y
    NEAR_FIELD_BOUNDS["y_max"] = args.near_field_y

    # ── Raw data ──────────────────────────────────────────────────────────
    print(f"Loading AirfRANS (task={args.task}) …")
    raw_train = AirfRANS(root=args.data_root, task=args.task, train=True)
    raw_test = AirfRANS(root=args.data_root, task=args.task, train=False)

    n_total = len(raw_train)
    val_size = max(1, int(n_total * 0.1))
    train_idx = list(range(n_total - val_size))
    val_idx = list(range(n_total - val_size, n_total))

    # ── Normalisation (must match training) ───────────────────────────────
    print("  Computing normalisation stats …")
    x_mean, x_std, p_mean, p_std = compute_normalisation(raw_train, train_idx)
    norm_stats = (x_mean, x_std, p_mean, p_std)

    # ── Datasets — sequential order (no shuffle) ──────────────────────────
    shared_cache: dict[int, GraphGrid] = {}
    train_ds = AirfRANSNearFieldDataset(
        args.data_root,
        "train",
        task=args.task,
        norm_stats=norm_stats,
        indices=train_idx,
        _shared_raw=raw_train,
        _shared_cache=shared_cache,
    )
    val_ds = AirfRANSNearFieldDataset(
        args.data_root,
        "train",
        task=args.task,
        norm_stats=norm_stats,
        indices=val_idx,
        _shared_raw=raw_train,
        _shared_cache=shared_cache,
    )
    test_ds = AirfRANSNearFieldDataset(
        args.data_root,
        "test",
        task=args.task,
        norm_stats=norm_stats,
        _shared_raw=raw_test,
    )
    n_train = len(train_ds)
    n_val = len(val_ds)
    n_test = len(test_ds)
    print(f"  {n_train} train / {n_val} val / {n_test} test")

    # ── Model ─────────────────────────────────────────────────────────────
    model = PressureDQEModel(**dims).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"  {sum(p.numel() for p in model.parameters()):,} parameters")

    # ── Pressure interval histograms + mean Cp ────────────────────────────
    print("  Collecting pressure interval histograms …")
    with torch.no_grad():
        embs_tr, pv_tr = collect_airfoil_embeddings(model, train_ds, device)
        embs_va, pv_va = collect_airfoil_embeddings(model, val_ds, device)
        embs_te, pv_te = collect_airfoil_embeddings(model, test_ds, device)

    all_embs = torch.cat([embs_tr, embs_va, embs_te]).numpy().astype(np.float64)
    all_pvals = torch.cat([pv_tr, pv_va, pv_te]).numpy()
    cp = all_pvals * p_std.item() + p_mean.item()  # de-normalised mean surface Cp

    # Pressure PCA → 2-D similarity coords
    E_c = all_embs - all_embs.mean(0)
    _, _, Vt_p = np.linalg.svd(E_c, full_matrices=False)
    p_coords = E_c @ Vt_p[:2].T  # (A, 2)

    # ── Shape vectors ─────────────────────────────────────────────────────
    # collect_shape_vectors iterates each dataset via the same _order that
    # collect_airfoil_embeddings used, so row i in shape == row i in embeddings.
    print("  Computing airfoil shape vectors …")
    shape_tr = collect_shape_vectors(raw_train, train_ds)
    shape_va = collect_shape_vectors(raw_train, val_ds)
    shape_te = collect_shape_vectors(raw_test, test_ds)
    all_shape = np.concatenate([shape_tr, shape_va, shape_te])  # (A, 2*N_CHORD)

    # Shape PCA → 2-D shape coords
    S_c = all_shape - all_shape.mean(0)
    _, _, Vt_s = np.linalg.svd(S_c, full_matrices=False)
    s_coords = S_c @ Vt_s[:2].T  # (A, 2)

    # Pearson r: pressure PC1 vs shape PC1
    r_val, p_val = pearsonr(p_coords[:, 0], s_coords[:, 0])
    print(f"  Pressure PC1 vs Shape PC1:  r = {r_val:+.3f}  (p = {p_val:.2e})")

    # ── Figure ────────────────────────────────────────────────────────────
    print("  Rendering figure …")

    xp, yp = p_coords[:, 0], p_coords[:, 1]
    xs, ys = s_coords[:, 0], s_coords[:, 1]

    # Interpolated pressure surface for the 3-D panel
    xi = np.linspace(xp.min(), xp.max(), 60)
    yi = np.linspace(yp.min(), yp.max(), 60)
    Xi, Yi = np.meshgrid(xi, yi)
    Zi = griddata((xp, yp), cp, (Xi, Yi), method="linear")
    Zi = np.where(np.isnan(Zi), np.nanmean(cp), Zi)

    # Shared Cp colour scale across both panels
    cp_min, cp_max = cp.min(), cp.max()

    fig = plt.figure(figsize=(19, 8))
    gs = gridspec.GridSpec(
        1,
        2,
        width_ratios=[1.45, 1],
        wspace=0.08,
        left=0.04,
        right=0.97,
        top=0.93,
        bottom=0.07,
    )

    # ── Left panel: 3-D pressure surface, points coloured by shape PC1 ────
    ax3 = fig.add_subplot(gs[0], projection="3d")

    ax3.plot_surface(
        Xi,
        Yi,
        Zi,
        cmap="RdBu_r",
        alpha=0.45,
        linewidth=0,
        antialiased=True,
        vmin=cp_min,
        vmax=cp_max,
    )

    # Scatter: colour = shape PC1 (viridis), size encodes nothing else
    shape_c = s_coords[:, 0]
    sc_shape = ax3.scatter(
        xp,
        yp,
        cp,
        c=shape_c,
        cmap="viridis",
        vmin=shape_c.min(),
        vmax=shape_c.max(),
        s=45,
        depthshade=True,
        edgecolors="k",
        linewidths=0.3,
        zorder=5,
    )
    cb_shape = fig.colorbar(
        sc_shape,
        ax=ax3,
        shrink=0.42,
        pad=0.04,
        label="Shape PC 1  (thickness/camber mode)",
    )

    ax3.set_xlabel("Pressure PC 1", labelpad=6)
    ax3.set_ylabel("Pressure PC 2", labelpad=6)
    ax3.set_zlabel("Mean surface Cp", labelpad=6)
    ax3.set_title(
        "Pressure-similarity colour = shape\n"
        f"Pressure–shape correlation:  r = {r_val:+.3f}",
        fontsize=10,
        pad=10,
    )

    # ── Right panel: 2-D shape space coloured by mean Cp ──────────────────
    ax2 = fig.add_subplot(gs[1])

    sc_cp = ax2.scatter(
        xs,
        ys,
        c=cp,
        cmap="RdBu_r",
        vmin=cp_min,
        vmax=cp_max,
        s=42,
        edgecolors="k",
        linewidths=0.3,
        alpha=0.88,
    )
    cb_cp = fig.colorbar(sc_cp, ax=ax2, shrink=0.78, label="Mean surface Cp")

    ax2.set_xlabel("Shape PC 1  (dominant thickness/camber mode)")
    ax2.set_ylabel("Shape PC 2")
    ax2.set_title("Shape similarity", fontsize=10)
    ax2.set_aspect("equal", adjustable="datalim")
    ax2.grid(True, lw=0.4, alpha=0.4)

    ax2.annotate(
        f"Pressure–shape\ncorrelation\nr = {r_val:+.3f}",
        xy=(0.04, 0.96),
        xycoords="axes fraction",
        va="top",
        ha="left",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#999", alpha=0.85),
    )

    # ── Save ──────────────────────────────────────────────────────────────
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


# ── CLI ───────────────────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Superimpose airfoil shape metric on 3-D pressure similarity surface",
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
    p.add_argument("--cpu", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    main(_parse_args())
