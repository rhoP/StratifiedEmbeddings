 Steel-Man

  The architecture has genuine intellectual merit.
  Stratifying a CFD mesh into regions with different
  geometric curvatures is principled: the boundary
  layer, wake, and freestream have fundamentally
  different turbulence structures. Letting each stratum
   learn its own curvature (hyperbolic for
  hierarchical/tree-like structure, spherical for
  periodic/symmetric regions, flat for baseline) rather
   than fixing it is a real contribution. The 4-phase
  training schedule carefully avoids the cold-start
  problem — warmup gives k-means something meaningful
  to cluster, k-means seeds the assigner with
  structure, the assigner pre-trains before the full
  geometry is live. The surface node weighting in
  airfrans_loss is domain-correct: aerodynamic
  coefficients are boundary integrals, not volume
  integrals.

  ---
  Concerns

  ---
  Concern: Phase 3 assigner objective is logically
  inverted — destroys k-means init
  Severity: Critical
  Framework: Confidence Without Correctness (AI Blind
  Spots #3)

  What I see:
    run_airfrans.py:379 — the assigner pre-training
  loss is:
        H = -(soft *
  soft.clamp(1e-8).log()).sum(-1).mean()
        loss = -H   # maximise entropy = uniform soft
  assignment
    The comment says "Assign each node to its k-means
  cluster as pseudo-label"
    but the code does the exact opposite: it trains the
   assigner to assign every
    node uniformly across ALL strata (maximum entropy =
   no preference for any
    stratum). This actively erases the cluster
  structure installed in Phase 2.

  Why it matters:
    Phase 2 runs k-means to initialize assigner
  centroids with semantic
    structure. Phase 3 then spends epochs training the
  assigner to forget
    that structure and output flat ~1/K distributions.
  Phase 4 starts from
    a destroyed initialization. The k-means step is
  wasted computation and
    gives a false sense of principled warm-starting.

  What to do:
    Replace the entropy objective with cross-entropy
  against k-means
    pseudo-labels. Pre-compute labels once from Phase
  2, then:
        ce_loss = F.cross_entropy(soft_logits,
  pseudo_labels[node_indices])
    where soft_logits = log(soft_assign) or the raw
  pre-softmax scores.
    That trains the assigner to reproduce the cluster
  structure instead of
    destroying it.

  ---
  Concern: Centripetal loss receives (N, d) where (K,
  d) expected — runtime crash
  Severity: Critical
  Framework: Pre-mortem

  What I see:
    run_airfrans.py:410-414 pre-computes
  expected_proto:
        protos_mean = protos_tan.mean(dim=1)      # (K,
   d)
        expected_proto = soft_assign @ protos_mean #
  (N, d)
    then passes expected_proto as protos_tan to
  total_loss().

    total_loss() (losses.py:116) passes protos_tan
  directly to:
        centripetal_loss(embeds_tan, protos_tan,
  soft_assign)

    centripetal_loss (losses.py:85) then does:
        expected_proto = soft_assign @ protos      #
  (N, K) @ (N, d) — SHAPE ERROR

    With N ≈ 8000 and K = 4: (8000, 4) @ (8000, 64)
  raises
    RuntimeError: mat1 and mat2 shapes cannot be
  multiplied.
    Phase 4 never runs.

  Why it matters:
    The code as written crashes on the first training
  step of Phase 4 —
    the primary training phase. If it were silently
  wrong instead of crashing
    (e.g., if N == K accidentally), the centripetal
  loss would be computing
    a double soft_assign application, pulling
  embeddings toward meaningless
    targets.

  What to do:
    Either (a) pass protos_mean (K, d) to total_loss
  instead of
    expected_proto, letting centripetal_loss do the
  expected-proto
    computation itself as designed; or (b) change
  centripetal_loss's
    signature to accept a pre-computed (N, d) target
  and drop the
    internal soft_assign @ protos line. Option (a) is
  the minimal fix.

  ---
  Concern: Node subsampling annihilates graph edges —
  GNN degrades to MLP
  Severity: High
  Framework: Inversion ("what would guarantee this
  fails?")

  What I see:
    subsample_nodes() (run_airfrans.py:119-140) draws
  max_nodes=8000 random
    nodes from ~180k, then filters edges to those with
  BOTH endpoints in the
    sample. In a KNN-8 graph, the probability both
  endpoints of a given edge
    are retained is (8000/180000)² ≈ 0.2%.

    At 8k nodes and ~1.44M edges (180k × 8), expected
  retained edges:
    1,440,000 × 0.0022 ≈ 3,168 — for 8,000 nodes.
    That's 0.4 edges per node on average. The GNN
  operates on an almost
    empty graph. SAGEConv with no neighbors collapses
  to its linear
    self-transform on each node independently — a
  per-node MLP.

  Why it matters:
    The entire inductive bias of the architecture —
  that spatial neighbors
    in the mesh carry aerodynamic context — is
  destroyed during training.
    The model learns to predict from local node
  features alone, then at
    evaluation (max_nodes=0, full graph) it's suddenly
  given dense
    neighborhood context it has never been trained to
  use.

  What to do:
    Use a neighbor-preserving sampling strategy. For
  GraphSAGE specifically,
    sample a subset of *graphs* (each AirfRANS sim is a
   separate graph) and
    train on full graphs when VRAM allows, or use PyG's
   NeighborLoader /
    ClusterData to sample subgraphs that preserve 2-hop
   connectivity.
    Alternatively, increase max_nodes significantly —
  at 50k nodes,
    edge retention rises to ~7.7%, still poor but far
  better.

  ---
  Concern: Geometry dispatch uses Python conditionals —
   no gradient through
           curvature type transitions
  Severity: High
  Framework: Socratic probing ("you're assuming
  gradients flow through κ")

  What I see:
    geometry.py:148-153 — exp_map_origin() does:
        if is_hyperbolic(kappa):     # calls
  float(kappa.item()) — detaches graph
            return hyp_exp_origin(v, K)
        if is_spherical(kappa):
            return sph_exp_origin(v, K)
        return v   # flat

    is_hyperbolic/is_spherical both call
  float(kappa.item()), which breaks
    the autograd graph. The *value* of kappa flows
  through the chosen branch's
    math (K = kappa.abs()), but the *sign decision*
  (which branch to execute)
    is invisible to autograd. If kappa starts near 0
  (flat), gradients from
    the flat path (identity: dL/d_kappa = 0 through the
   map itself) cannot
    push kappa into a non-flat regime. Kappa can get
  stuck at flat indefinitely.

  Why it matters:
    The system's core theoretical claim is that strata
  *learn* their geometry
    type. But a stratum initialized near zero may never
   transition to
    hyperbolic or spherical because the gradient
  landscape at the transition
    boundary has a discontinuity that autograd cannot
  navigate. Diversity loss
    can push kappa values apart in magnitude, but if
  most start near 0, they
    all push each other into the flat regime
  symmetrically.

  What to do:
    Use smooth interpolation between geometry types.
  One approach: compute
    all three maps (hyp, flat, sph) and blend by
  sigmoid gates derived from
    kappa. This is differentiable through the geometry
  type. Alternatively,
    acknowledge that geometry type is discrete and use
  straight-through
    estimator (STE) for the branch selection.

  ---
  Concern: Interval boundary parameters unconstrained —
   intervals silently die
  Severity: Medium
  Framework: Blind spots — edge cases

  What I see:
    stratified_dqe.py:194-195 — interval boundaries are
   plain nn.Parameter:
        self.interval_lo =
  nn.Parameter(torch.linspace(-2, 2, n_intervals))
        self.interval_hi =
  nn.Parameter(torch.linspace(-1, 3, n_intervals))
    There is no constraint enforcing interval_lo[i] <
  interval_hi[i].
    The interval weight formula (line 243-244) is:
        σ((x - lo)/t) * σ((hi - x)/t)
    When lo ≥ hi, both sigmoids are simultaneously <
  0.5 for all x, so
    the weight is globally near zero. After
  normalization (line 245), that
    interval's weight gets dominated by noise. No loss
  term penalizes this.

  Why it matters:
    A dead interval contributes nothing to the quotient
   vector but still
    occupies columns in the head weight matrix. The
  effective capacity of
    the DQE shrinks silently. With 8 intervals, losing
  2-3 means the model
    has fewer degrees of freedom than it appears to
  have. The code gives no
    diagnostic for this — dead intervals are invisible
  in the logged losses.

  What to do:
    Parameterize as center + positive width: lo =
  center - exp(log_half_width),
    hi = center + exp(log_half_width). This guarantees
  lo < hi by construction
    and doesn't require any constraint projection.
  Alternatively, add a hinge
    penalty: sum(relu(lo - hi + margin)).

  ---
  Concern: Checkpoint selected on degraded (subsampled)
   validation metric
  Severity: Medium
  Framework: Pre-mortem

  What I see:
    During Phase 4, the best checkpoint is selected by:
        val_metrics = evaluate(model, val_list, ...,
  max_nodes=args.max_nodes)
    using the same 8k-node subsampling (and thus ~0.4
  edges/node) as training.
    But the final test evaluation at
  run_airfrans.py:471 uses max_nodes=0
    (full graph, dense edges). The model is selected
  for performance on a
    nearly edge-free graph, then evaluated on a
  fully-connected mesh.

  Why it matters:
    Checkpoint selection optimizes for the wrong
  distribution. The "best"
    epoch by validation MSE is the epoch that learned
  most effectively
    without neighborhoods — which may not be the epoch
  that performs best
    with full connectivity. In the worst case, the
  model overfits to the
    MLPbehavior and the checkpoint that generalizes
  best on the full graph
    is discarded.

  What to do:
    Run validation on full graphs (max_nodes=0) or at
  minimum on a much
    larger subsample (≥50k nodes). If GPU memory is the
   constraint, move
    validation to CPU or run it on a single
  representative sample with the
    full graph rather than the subsampled version.

  ---
  Concern: No incentive for strata to specialize —
  entropy regularizer
           rewards the degenerate uniform-mixing
  solution
  Severity: Medium
  Framework: Inversion

  What I see:
    The entropy loss (losses.py:37-51) maximises
  per-node assignment entropy.
    Maximum entropy means every node assigns equal
  weight 1/K to all strata.
    At the limit: soft_assign = [[1/K, 1/K, ...]] for
  all nodes.

    At this limit: stratum_entropy_loss = 0 (optimal),
  curvature_diversity
    can push kappas apart (satisfying that term), and
  the regression loss
    is just Σ_k (1/K) * pred_k(embed) — a simple
  average of K independent
    heads. If the heads learn the same function, the
  model is a single
    predictor with K redundant copies. Nothing in the
  loss landscape forces
    the strata to learn *different* functions.

  Why it matters:
    The hypothesis that different geometric strata
  capture different flow
    regimes is the scientific contribution of this
  model. But the loss
    does not operationalize this hypothesis. A
  degenerate collapse where
    all strata predict identically with uniform mixing
  achieves low
    regression loss, zero entropy loss, and can satisfy
   diversity loss
    (κ values differ, but the functions are the same).
  The geometry is
    decorative.

  What to do:
    Add a stratum specialization incentive. Options:
  (1) add a prediction
    diversity loss penalizing when pred_k ≈ pred_j for
  k ≠ j; (2) use
    winner-take-all hard routing during some fraction
  of training steps
    (straight-through); (3) use a conditional entropy
  term that rewards
    low assignment entropy *and* high prediction
  diversity jointly.

  ---
  Verdict: Rethink this

  The architecture is intellectually compelling and the
   domain-specific choices (SAGEConv for inductive mesh
   inference, surface weighting, curvature per stratum)
   are sound. But two issues make Phase 4 currently
  non-functional: the assigner pre-training reverses
  its own initialization, and the centripetal loss has
  a shape mismatch that crashes on the first Phase 4
  step. Fix those two and the code runs. Then the
  deeper structural issues — subsampled training
  destroying graph connectivity, and no force for
  strata to actually specialize — determine whether the
   model learns what you think it's learning.

  Priority order for fixes: #2 (crash) → #1 (inverted
  Phase 3) → #3 (subsampling) → the rest.
