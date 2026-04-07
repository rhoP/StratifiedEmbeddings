"""
StratifiedDQE — Differentiable Quotient Embedding in a stratified space.

Architecture
------------
                      ┌─────────────────────────────────┐
   node features  →   │  NodeGNNEncoder (SAGEConv)       │  → shared embed (N, d)
                      └─────────────────────────────────┘
                                      │
                    ┌─────────────────┴──────────────────┐
                    ▼                                     ▼
          StratumAssigner                      per-stratum StratumDQE[k]
          (N, K) soft weights        for each stratum k:
                    │                  embed  →  exp_map(κ_k)
                    │                  → interval soft-assign
                    │                  → prototype soft-assign
                    │                  → outer-product quotient vec
                    └────────────────────┐
                                         ▼
                               Shared regression head
                               (N, n_targets) predictions

Training phases
---------------
Phase 1 — Warmup regressor
    Train NodeGNNEncoder + WarmupHead with plain L2 regression.

Phase 2 — K-means init
    Cluster the frozen encoder outputs into K strata.
    Initialise StratumAssigner centroids and DQE prototypes from cluster centres.

Phase 3 — Stratum assigner pre-train
    Freeze encoder; train StratumAssigner to predict cluster labels.

Phase 4 — Full joint training
    All parameters unfrozen.  Loss = regression + entropy + diversity + centripetal.
    After each epoch: update hard stratum assignments for logging / re-init.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

from .geometry import (
    exp_map_origin,
    log_map_origin,
    clip_to_manifold,
    dist_to_protos,
    EPS,
)

# ── Hyper-constants ─────────────────────────────────────────────────────────

K_MAX_DEFAULT = 2.0     # |κ| bounded to (-K_max, +K_max) via tanh


# ── Encoder ─────────────────────────────────────────────────────────────────

class NodeGNNEncoder(nn.Module):
    """GraphSAGE encoder producing per-node embeddings.

    SAGEConv is chosen for AirfRANS because:
    - Efficient on large, dense meshes (no attention, O(E) work).
    - Mean aggregation is robust to varying local node degrees in KNN graphs.
    - Inductive: can encode unseen geometries at test time.
    """

    def __init__(
        self,
        in_dim:     int,
        hidden_dim: int,
        embed_dim:  int,
        n_layers:   int = 4,
        dropout:    float = 0.1,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.convs  = nn.ModuleList()
        self.norms  = nn.ModuleList()
        for i in range(n_layers):
            in_c  = hidden_dim
            out_c = hidden_dim if i < n_layers - 1 else embed_dim
            self.convs.append(SAGEConv(in_c, out_c))
            self.norms.append(nn.LayerNorm(out_c))
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x:          torch.Tensor,   # (N, in_dim)
        edge_index: torch.Tensor,   # (2, E)
    ) -> torch.Tensor:              # (N, embed_dim)
        h = self.input_proj(x)
        for conv, norm in zip(self.convs, self.norms):
            h = norm(F.silu(conv(h, edge_index)))
            h = self.dropout(h)
        return h


# ── Warmup head ──────────────────────────────────────────────────────────────

class WarmupRegressor(nn.Module):
    """Simple MLP regression head used only during Phase 1 warmup."""

    def __init__(self, embed_dim: int, n_targets: int, hidden: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, n_targets),
        )

    def forward(self, embeds: torch.Tensor) -> torch.Tensor:
        return self.net(embeds)


# ── Stratum assigner ─────────────────────────────────────────────────────────

class StratumAssigner(nn.Module):
    """Soft assignment of each node to one of K strata.

    Assignments = softmax(-T · dist(embed, centroid_k))

    where T is a learnable temperature (log-parameterised for positivity)
    and centroids are learnable in tangent / Euclidean space.
    """

    def __init__(self, embed_dim: int, n_strata: int) -> None:
        super().__init__()
        self.n_strata = n_strata
        self.centroids = nn.Parameter(torch.randn(n_strata, embed_dim) * 0.1)
        self.log_temp  = nn.Parameter(torch.zeros(()))  # temperature = exp(log_temp)

    @property
    def temperature(self) -> torch.Tensor:
        return self.log_temp.exp().clamp(min=1e-2, max=100.0)

    def forward(self, embeds: torch.Tensor) -> torch.Tensor:
        """Return soft assignments (N, K) summing to 1."""
        # Euclidean L2 distance in tangent space
        diff = embeds.unsqueeze(1) - self.centroids.unsqueeze(0)  # (N, K, d)
        sq_dist = (diff ** 2).sum(-1)                              # (N, K)
        return F.softmax(-self.temperature * sq_dist, dim=-1)      # (N, K)

    def hard_assignments(self, embeds: torch.Tensor) -> torch.Tensor:
        """Return argmax stratum index per node (N,)."""
        return self.forward(embeds).argmax(dim=-1)

    def init_centroids(self, cluster_centres: torch.Tensor) -> None:
        """Initialise from k-means centroids (K, d)."""
        with torch.no_grad():
            self.centroids.copy_(cluster_centres)


# ── Per-stratum DQE ──────────────────────────────────────────────────────────

class StratumDQE(nn.Module):
    """DQE map for a single stratum with learnable curvature κ.

    Procedure
    ---------
    1. Map embed → manifold point via exp_map_origin(·, κ).
    2. Assign to n_intervals soft intervals along the manifold via
       double-sigmoid windows applied to a 1-D projection.
    3. Assign to n_protos soft prototypes via softmax over geodesic distances.
    4. Form outer-product quotient vector: (n_intervals × n_protos,).
    5. Linear decode to n_targets.
    """

    def __init__(
        self,
        embed_dim:   int,
        n_intervals: int,
        n_protos:    int,
        n_targets:   int,
        K_max:       float = K_MAX_DEFAULT,
        interval_temp: float = 0.5,
        proto_temp:    float = 1.0,
    ) -> None:
        super().__init__()
        self.embed_dim    = embed_dim
        self.n_intervals  = n_intervals
        self.n_protos     = n_protos
        self.n_targets    = n_targets
        self.K_max        = K_max

        # Learnable curvature (unconstrained → bounded via tanh)
        self.kappa_raw = nn.Parameter(torch.zeros(()))

        # Interval boundaries parameterised as centre ± exp(log_half_width).
        # This guarantees lo < hi by construction — gradients can never flip
        # the ordering, so intervals cannot silently die during training.
        # Initialised to match the original linspace(-2,2) / linspace(-1,3):
        #   centre = (-1.5 … 2.5), half_width = 0.5 uniformly.
        self.interval_center  = nn.Parameter(torch.linspace(-1.5, 2.5, n_intervals))
        self.log_half_width   = nn.Parameter(torch.full((n_intervals,), math.log(0.5)))
        self.log_interval_temp = nn.Parameter(torch.full((), math.log(interval_temp)))
        self.proj_weight = nn.Parameter(F.normalize(torch.randn(embed_dim), dim=0))

        # Prototypes live in tangent space; mapped to manifold on demand
        self.protos_tan  = nn.Parameter(torch.randn(n_protos, embed_dim) * 0.1)
        self.log_proto_temp = nn.Parameter(torch.full((), math.log(proto_temp)))

        # Output head
        self.head = nn.Linear(n_intervals * n_protos, n_targets)

    @property
    def kappa(self) -> torch.Tensor:
        return self.K_max * torch.tanh(self.kappa_raw)

    @property
    def interval_lo(self) -> torch.Tensor:
        return self.interval_center - self.log_half_width.exp()

    @property
    def interval_hi(self) -> torch.Tensor:
        return self.interval_center + self.log_half_width.exp()

    @property
    def interval_temp(self) -> torch.Tensor:
        return self.log_interval_temp.exp().clamp(min=1e-3)

    @property
    def proto_temp(self) -> torch.Tensor:
        return self.log_proto_temp.exp().clamp(min=1e-3)

    def forward(
        self,
        embeds: torch.Tensor,   # (N, d) — tangent-space encoder output
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        pred       : (N, n_targets)
        interval_w : (N, n_intervals)  — for logging / centripetal loss
        proto_w    : (N, n_protos)     — for logging / centripetal loss
        """
        kappa = self.kappa

        # 1 — map to manifold
        pts = exp_map_origin(embeds, kappa)        # (N, d)
        pts = clip_to_manifold(pts, kappa)

        # 2 — soft interval assignment via 1-D projection
        x_tan = log_map_origin(pts, kappa)         # back to tangent for projection
        w_hat = F.normalize(self.proj_weight.unsqueeze(0), dim=-1)  # (1, d)
        scalar = (x_tan.float() @ w_hat.T).squeeze(-1)  # (N,)

        lo = self.interval_lo  # (I,)
        hi = self.interval_hi  # (I,)
        t  = self.interval_temp
        iv_w = torch.sigmoid((scalar.unsqueeze(-1) - lo) / t) \
             * torch.sigmoid((hi - scalar.unsqueeze(-1)) / t)   # (N, I)
        iv_w = iv_w / iv_w.sum(dim=-1, keepdim=True).clamp(min=EPS)

        # 3 — soft prototype assignment via geodesic distances
        protos_mfld = exp_map_origin(self.protos_tan, kappa)  # (P, d)
        protos_mfld = clip_to_manifold(protos_mfld, kappa)
        dists = dist_to_protos(pts, protos_mfld, kappa)       # (N, P)
        pr_w  = F.softmax(-dists / self.proto_temp, dim=-1)   # (N, P)

        # 4 — outer product quotient vector
        quotient = torch.bmm(
            iv_w.unsqueeze(-1),   # (N, I, 1)
            pr_w.unsqueeze(-2),   # (N, 1, P)
        ).reshape(embeds.shape[0], -1)                          # (N, I*P)

        # 5 — decode
        pred = self.head(quotient.to(self.head.weight.dtype))  # (N, T)
        return pred, iv_w, pr_w


# ── Full StratifiedDQE ───────────────────────────────────────────────────────

class StratifiedDQE(nn.Module):
    """Joint model: shared encoder + stratum assigner + K per-stratum DQEs.

    Forward pass
    ------------
    embed   = encoder(x, edge_index)           (N, d)
    weights = assigner(embed)                   (N, K)  soft assignments
    pred_k  = dqe_k(embed)  for each k          (N, T) per stratum
    pred    = Σ_k weights[:,k:k+1] * pred_k    (N, T)  weighted mixture
    """

    def __init__(
        self,
        in_dim:        int,
        hidden_dim:    int,
        embed_dim:     int,
        n_strata:      int,
        n_intervals:   int,
        n_protos:      int,
        n_targets:     int,
        n_layers:      int = 4,
        dropout:       float = 0.1,
        K_max:         float = K_MAX_DEFAULT,
        interval_temp: float = 0.5,
        proto_temp:    float = 1.0,
    ) -> None:
        super().__init__()
        self.n_strata   = n_strata
        self.embed_dim  = embed_dim

        self.encoder  = NodeGNNEncoder(in_dim, hidden_dim, embed_dim, n_layers, dropout)
        self.assigner = StratumAssigner(embed_dim, n_strata)
        self.dqes     = nn.ModuleList([
            StratumDQE(embed_dim, n_intervals, n_protos, n_targets,
                       K_max, interval_temp, proto_temp)
            for _ in range(n_strata)
        ])
        self.warmup_head = WarmupRegressor(embed_dim, n_targets)

    # ------------------------------------------------------------------
    # Phase helpers
    # ------------------------------------------------------------------

    def set_phase(self, phase: int) -> None:
        """Freeze / unfreeze parameter groups per training phase.

        Phase 1 — warmup: only encoder + warmup_head
        Phase 2 — assigner pre-train: only assigner (encoder frozen)
        Phase 3 — full: everything unfrozen, warmup_head frozen
        """
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

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def encode(
        self,
        x:          torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        return self.encoder(x, edge_index)

    def warmup_forward(
        self,
        x:          torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """Phase 1: plain regression without DQE."""
        embed = self.encode(x, edge_index)
        return self.warmup_head(embed)

    def forward(
        self,
        x:          torch.Tensor,
        edge_index: torch.Tensor,
        use_wta:    bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full forward pass (Phases 3+).

        Parameters
        ----------
        use_wta : bool
            When True, routes predictions through a hard winner-take-all
            assignment with a straight-through gradient estimator (option 2
            specialization).  Each node is assigned to its highest-weight
            stratum in the forward pass; gradients flow through soft_assign
            in the backward pass.  soft_assign is always returned unmodified
            so other loss terms (entropy, centripetal) still see soft weights.

        Returns
        -------
        pred        : (N, n_targets)
        soft_assign : (N, K)            — soft weights, independent of use_wta
        embeds      : (N, d)            — tangent-space encoder output
        protos_tan  : (K, n_protos, d)  — stratum prototype tangent coords
        kappas      : (K,)              — one learnable κ per stratum
        preds_stack : (N, K, n_targets) — per-stratum predictions before mixing
        """
        embeds      = self.encode(x, edge_index)            # (N, d)
        soft_assign = self.assigner(embeds)                  # (N, K)

        preds:  list[torch.Tensor] = []
        kappas: list[torch.Tensor] = []
        for dqe in self.dqes:
            assert isinstance(dqe, StratumDQE)
            p, _, _ = dqe(embeds)                            # (N, T)
            preds.append(p)
            kappas.append(dqe.kappa)

        preds_stack = torch.stack(preds, dim=1)              # (N, K, T)

        if use_wta:
            # Straight-through estimator: hard one-hot routing in the forward
            # pass, gradient through soft_assign in the backward pass.
            hard_idx     = soft_assign.detach().argmax(dim=-1)          # (N,)
            hard_weights = F.one_hot(hard_idx, self.n_strata).to(soft_assign.dtype)
            routing = hard_weights + soft_assign - soft_assign.detach() # (N, K)
        else:
            routing = soft_assign

        pred = (routing.unsqueeze(-1) * preds_stack).sum(dim=1)         # (N, T)

        kappas_t = torch.stack(kappas)                       # (K,)

        # Tangent-space prototypes for centripetal loss: (K, P, d) stacked
        protos_tan = torch.stack(
            [dqe.protos_tan for dqe in self.dqes if isinstance(dqe, StratumDQE)]
        )  # (K, P, d)

        return pred, soft_assign, embeds, protos_tan, kappas_t, preds_stack

    # ------------------------------------------------------------------
    # Initialisation from k-means
    # ------------------------------------------------------------------

    @torch.no_grad()
    def init_from_kmeans(
        self,
        cluster_centres: torch.Tensor,  # (K, d)
        assign_labels:   torch.Tensor,  # (N,) hard labels used to init proto
        all_embeds:      torch.Tensor,  # (N, d) encoder outputs
    ) -> None:
        """Warm-start assigner centroids and DQE prototypes."""
        K = self.n_strata
        self.assigner.init_centroids(cluster_centres)

        dqe_list: list[StratumDQE] = [m for m in self.dqes if isinstance(m, StratumDQE)]
        for k, dqe in enumerate(dqe_list):
            mask = assign_labels == k
            if mask.sum() == 0:
                continue
            cluster_embeds = all_embeds[mask]                     # (Nk, d)
            P = dqe.n_protos
            # sub-sample P points as proto initialisation
            idx = torch.randperm(cluster_embeds.shape[0])[:P]
            if idx.numel() > 0:
                n = idx.numel()
                dqe.protos_tan.data[:n] = cluster_embeds[idx]
