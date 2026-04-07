"""
StratifiedEmbedding — Differentiable Quotient Embeddings in stratified spaces.
"""

from .geometry import (
    geometry_name,
    is_hyperbolic,
    is_spherical,
    exp_map_origin,
    log_map_origin,
    clip_to_manifold,
    tangent_proj,
    dist_to_protos,
)

from .losses import (
    airfrans_loss,
    stratum_entropy_loss,
    curvature_diversity_loss,
    centripetal_loss,
    total_loss,
    prediction_diversity_loss,
    conditional_diversity_loss,
)

from .stratified_dqe import (
    NodeGNNEncoder,
    WarmupRegressor,
    StratumAssigner,
    StratumDQE,
    StratifiedDQE,
)

from .viz_stratified import (
    plot_training_curves,
    plot_curvature_evolution,
    plot_stratum_assignments,
    plot_prediction_scatter,
    plot_geometry_summary,
)

__all__ = [
    # geometry
    "geometry_name", "is_hyperbolic", "is_spherical",
    "exp_map_origin", "log_map_origin", "clip_to_manifold",
    "tangent_proj", "dist_to_protos",
    # losses
    "airfrans_loss", "stratum_entropy_loss", "curvature_diversity_loss",
    "centripetal_loss", "total_loss",
    "prediction_diversity_loss", "conditional_diversity_loss",
    # model
    "NodeGNNEncoder", "WarmupRegressor", "StratumAssigner",
    "StratumDQE", "StratifiedDQE",
    # viz
    "plot_training_curves", "plot_curvature_evolution",
    "plot_stratum_assignments", "plot_prediction_scatter",
    "plot_geometry_summary",
]
