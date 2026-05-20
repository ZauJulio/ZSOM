"""Self-Organizing Map (SOM) core implementation.

A SOM is an unsupervised neural network that learns a low-dimensional (typically 2D)
discrete representation of high-dimensional input space, preserving topological structure.

Mathematical foundation
-----------------------
Given a dataset X ∈ ℝ^(n×d) and a weight matrix W ∈ ℝ^(w×h×d), training iterates:

    1. BMU selection:  bmu = argmin_{i,j} ‖x - W_{i,j}‖
    2. Neighbourhood:  h(i,j) = exp(−d_grid(bmu, (i,j))² / (2σ²))
    3. Weight update:  ΔW_{i,j} = η · h(i,j) · (x − W_{i,j})

where η is the learning rate and σ is the neighborhood radius,
both decaying monotonically over epochs.

Real-world applications
-----------------------
- **Customer segmentation**: Cluster customers by purchase behavior and project
  them onto a 2D map where neighboring regions share spending patterns.
- **Bioinformatics**: Visualize gene expression profiles across thousands of genes;
  co-expressed genes form spatial clusters on the map.
- **Anomaly detection**: Inputs that land far from any BMU (high quantization error)
  are likely outliers — useful in fraud detection or sensor fault diagnosis.
- **NLP / document clustering**: Project TF-IDF or embedding vectors onto a grid
  to visualize topic clusters across a document corpus.
"""
from __future__ import annotations

from typing import Literal

import numpy as np

from .metrics import DistanceFunc, MetricName, METRICS, dist_euclidean
from .numba_accel import HAS_NUMBA, batch_update_core, euclidean_distances_batch

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

Topology = Literal["square", "hex"]
Decay_Schedule = Literal["linear", "exponential"]

"""Valid grid topologies.

``'square'``
    Each node has up to 4 cardinal neighbours (von Neumann neighbourhood).
    Standard choice for most tasks; faster to compute grid distances.

``'hex'``
    Each node has up to 6 neighbours on a hexagonal offset grid.
    Provides more uniform spatial coverage and smoother U-Matrix boundaries,
    at the cost of slightly higher computation.

Example — choosing topology
    A square grid works well for a 20×20 customer-segmentation map.
    A hex grid is preferred when the data manifold has curved or circular
    structure (e.g. colour wheel, cyclic time-series).
"""

InitMethod = Literal["random", "pca"]
"""Weight initialisation strategy.

``'random'``
    Weights drawn uniformly from [0, 1). Fast, but may require more epochs to
    converge. Good default when data distribution is unknown or non-linear.

``'pca'``
    Weights initialised along the first two principal components of the data.
    Deterministic and typically halves the number of epochs needed.
    Preferred for tabular data (e.g. financial features, sensor readings).

Mathematical note
    PCA init spans the subspace defined by eigenvectors v₁, v₂ of the
    data covariance matrix Σ = (1/n) Xᵀ X. The grid is a linear lattice
    in that 2D plane, projected back to ℝ^d.
"""


# ---------------------------------------------------------------------------
# Hex grid utilities
# ---------------------------------------------------------------------------

def hex_grid_coords(w: int, h: int) -> np.ndarray:
    """Compute Cartesian (x, y) positions for nodes in a hexagonal offset grid.

    In a hex grid, odd-indexed columns (``j % 2 == 1``) are offset by 0.5
    along the x-axis, and rows are spaced by √3/2 ≈ 0.866 to maintain
    unit-distance adjacency between all 6 neighbors.

    Layout (j=0 even, j=1 odd)::

        col 0   col 1   col 2
        (0,0)   (0,1)   (0,2)
          (1,0)   (1,1)   (1,2)
        (2,0)   (2,1)   ...

    Parameters
    ----------
    w : int
        Number of columns in the grid.
    h : int
        Number of rows in the grid.

    Returns
    -------
    np.ndarray, shape (w, h, 2)
        ``coords[i, j]`` is the (x, y) Cartesian position of node (i, j).

    Example
    -------
    For a 10×10 hex SOM trained on RGB colour data, ``hex_grid_coords``
    ensures that perceptually similar colors (e.g. shades of blue) map
    to spatially adjacent nodes with equal inter-node distances in all
    six directions, avoiding the directional bias of square grids.
    """
    ii, jj = np.meshgrid(
        np.arange(w, dtype=np.float64),
        np.arange(h, dtype=np.float64),
        indexing="ij",
    )
    coords = np.empty((w, h, 2))
    coords[..., 0] = ii + 0.5 * (jj % 2)       # offset odd rows
    coords[..., 1] = jj * (np.sqrt(3) / 2.0)
    return coords


# ---------------------------------------------------------------------------
# PCA initialization helper
# ---------------------------------------------------------------------------

def _pca_init_weights(
    w: int,
    h: int,
    input_dim: int,
    data: np.ndarray,
) -> np.ndarray:
    """Initialise SOM weights along the first two principal components of ``data``.

    Algorithm
    ---------
    1. Centre the data: X̃ = X − μ, where μ = (1/n) Σ xᵢ.
    2. Compute the compact SVD: X̃ = U Σ Vᵀ.
       - Rows of Vᵀ are the principal directions v₁, v₂ … ∈ ℝ^d.
    3. Project data onto the top-2 PCs: S = X̃ Vᵀ[:2]ᵀ ∈ ℝ^(n×2).
    4. Build a (w × h) lattice spanning [min(S₁), max(S₁)] × [min(S₂), max(S₂)].
    5. Map each lattice point back to ℝ^d: W = μ + grid_scores · Vᵀ[:2].

    This ensures the initial weight surface covers the dominant variance
    directions of the data, dramatically reducing the number of epochs
    needed for convergence.

    Parameters
    ----------
    w : int
        Grid width (number of columns).
    h : int
        Grid height (number of rows).
    input_dim : int
        Dimensionality d of each input vector.
    data : np.ndarray, shape (n, d)
        Training samples used to compute principal components.
        Should be the same dataset passed to :meth:`SOM.fit`.

    Returns
    -------
    np.ndarray, shape (w, h, d)
        Initial weight matrix aligned with the data manifold.

    Notes
    -----
    Falls back gracefully when ``input_dim == 1``: uses a single PC and
    sets the second dimension of the grid to a constant zero projection.

    Example
    -------
    Suppose your dataset contains 10,000 customer records with 50 features
    (age, income, purchase frequency, …). PCA init aligns the SOM's first
    axis with the direction of maximum variance (e.g. high-vs-low income)
    and the second axis with the next most explanatory direction
    (e.g. young frequent buyer vs. older occasional buyer). This means the
    map already encodes the dominant structure *before* training begins.
    """
    # Center the data
    mean = data.mean(axis=0)
    centered = data - mean

    # SVD to get principal components (no sklearn needed)
    # U: (n, n), S: (min(n,d),), Vt: (d, d)
    _, S, Vt = np.linalg.svd(centered, full_matrices=False)

    # First 2 principal components (rows of Vt)
    n_components = min(2, input_dim)
    pc = Vt[:n_components]  # (n_components, input_dim)

    # Project data onto PCs to get ranges
    scores = centered @ pc.T  # (n, n_components)
    mins = scores.min(axis=0)
    maxs = scores.max(axis=0)

    # Create a linearly spaced grid in PC space
    if n_components >= 2:
        g1 = np.linspace(mins[0], maxs[0], w)
        g2 = np.linspace(mins[1], maxs[1], h)
        grid_pc1, grid_pc2 = np.meshgrid(g1, g2, indexing="ij")
        grid_scores = np.stack([grid_pc1, grid_pc2], axis=-1)  # (w, h, 2)
    else:
        g1 = np.linspace(mins[0], maxs[0], w)
        grid_scores = np.zeros((w, h, 1))
        for j in range(h):
            grid_scores[:, j, 0] = g1

    # Project back to original space
    weights = np.full((w, h, input_dim), mean)
    weights += (grid_scores @ pc[:n_components])  # (w, h, input_dim)
    return weights


# ---------------------------------------------------------------------------
# SOM
# ---------------------------------------------------------------------------

class SOM:
    """A 2D Self-Organizing Map with configurable topology and distance metric.

    A SOM learns a topology-preserving mapping f: ℝ^d → ℤ²  from
    high-dimensional input space to a discrete 2D grid of prototype vectors
    (weights). After training, nearby grid nodes represent similar inputs.

    Grid dimensions
    ---------------
    A common heuristic is ``w × h ≈ 5 × √n``, where n is the number of
    training samples. For example, with 10,000 samples a 50×50 grid
    (~2,500 nodes) is reasonable. Larger grids capture finer structure
    but require more epochs and memory.

    Topology comparison
    -------------------
    +----------+--------------------------------------------+
    | square   | Faster; directional bias along axes        |
    | hex      | More isotropic; better for curved manifolds|
    +----------+--------------------------------------------+

    Metrics
    -------
    The distance metric determines BMU selection and, indirectly, what the
    SOM learns to represent:

    - ``'euclidean'`` (default): sensitive to magnitude differences.
      Good for continuous sensor data, financial time-series.
    - ``'manhattan'``:  L1 norm, more robust to outliers.
      Useful for sparse count data (word frequencies, event logs).
    - ``'cosine'``: angle between vectors, ignores magnitude.
      Preferred for NLP embeddings and normalized feature vectors.

    Parameters
    ----------
    width : int
        Number of columns w in the 2D grid.
    height : int
        Number of rows h in the 2D grid.
    input_dim : int
        Dimensionality d of each input sample.
    rng : np.random.Generator
        NumPy random generator for reproducible weight initialization.
        Create with ``np.random.default_rng(seed)``.
    topology : Topology, default ``'square'``
        Grid connectivity. See :data:`Topology` for details.
    metric : DistanceFunc | MetricName | None, default ``dist_euclidean``
        Distance function for BMU search. Pass a string (``'euclidean'``,
        ``'manhattan'``, ``'cosine'``) or a callable
        ``(weights: ndarray(w,h,d), point: ndarray(d)) -> ndarray(w,h)``.
    use_numba : bool | None, default ``None``
        ``None`` → auto-detect: enables Numba JIT when available and
        conditions are met (square topology + euclidean metric).
        ``True`` → force Numba (raises if not available).
        ``False`` → always use NumPy fallback.
    init : InitMethod, default ``'random'``
        Weight initialization strategy. ``'pca'`` requires ``data``.
    data : np.ndarray | None
        Training data required when ``init='pca'``.

    Attributes
    ----------
    weights : np.ndarray, shape (w, h, d)
        Current prototype weight matrix. After training, ``weights[i, j]``
        is the learned centroid of all inputs that map to node (i, j).
    error_history_ : list[float]
        Quantization error (QE) recorded after each training epoch.
        Use this to monitor convergence: a flattening curve indicates
        the SOM has stabilized.
    hex_coords : np.ndarray | None, shape (w, h, 2)
        Cartesian grid positions (hex topology only).

    Example — customer segmentation
    --------------------------------
    >>> import numpy as np
    >>> from zsom import SOM
    >>>
    >>> # 5,000 customers × 10 behavioural features
    >>> data = np.random.default_rng(0).standard_normal((5000, 10))
    >>> rng  = np.random.default_rng(42)
    >>>
    >>> som = SOM(20, 20, input_dim=10, rng=rng, init='pca', data=data)
    >>> snapshots = som.fit(data, epochs=50, learning_rate=0.5)
    >>>
    >>> # Map each customer to a 2D grid position
    >>> bmus = som.get_bmus(data)        # shape (5000, 2)
    >>> u    = som.get_u_matrix()        # shape (20, 20) — visualise clusters
    """

    def __init__(
        self,
        width: int,
        height: int,
        input_dim: int,
        rng: np.random.Generator,
        topology: Topology = "square",
        decay_schedule: Decay_Schedule = "linear",
        metric: DistanceFunc | MetricName | None = None,
        use_numba: bool | None = None,
        init: InitMethod = "random",
        data: np.ndarray | None = None,
    ) -> None:
        """Initialise the SOM. See class docstring for parameter details."""
        if topology not in ("square", "hex"):
            raise ValueError(
                f"topology must be 'square' or 'hex', got {topology!r}"
            )
        self.w = width
        self.h = height
        self.input_dim = input_dim
        self.topology: Topology = topology
        if decay_schedule not in ("linear", "exponential"):
            raise ValueError(
                f"decay_schedule must be 'linear' or 'exponential', got {decay_schedule!r}"
            )
            
        self.decay_schedule: Literal["linear", "exponential"] = decay_schedule

        if isinstance(metric, str):
            if metric not in METRICS:
                raise ValueError(f"Unknown metric {metric!r}. Available: {list(METRICS)}")
            self.metric = METRICS[metric]  # ty:ignore[invalid-argument-type]
        else:
            self.metric = metric if metric is not None else dist_euclidean

        self.error_history_: list[float] = []

        # --- Weight initialization ---
        if init == "pca":
            if data is None:
                raise ValueError("init='pca' requires the 'data' parameter")
            self.weights: np.ndarray = _pca_init_weights(width, height, input_dim, data)
        else:
            self.weights: np.ndarray = rng.random((width, height, input_dim))

        self.hex_coords: np.ndarray | None = (
            hex_grid_coords(width, height) if topology == "hex" else None
        )

        # Numba: auto-detect or honor explicit flag
        if use_numba is None:
            self.use_numba: bool = (
                HAS_NUMBA
                and topology == "square"
                and self.metric is dist_euclidean
            )
        else:
            self.use_numba = use_numba and HAS_NUMBA

        # Precompute (w, h, w, h) grid distance table for square topology
        if topology == "square":
            ii = np.arange(width)[:, None, None, None]
            jj = np.arange(height)[None, :, None, None]
            bi = np.arange(width)[None, None, :, None]
            bj = np.arange(height)[None, None, None, :]
            self._gdist: np.ndarray | None = np.sqrt(
                (ii - bi) ** 2 + (jj - bj) ** 2
            )
        else:
            self._gdist = None

    # ------------------------------------------------------------------
    # BMU search
    # ------------------------------------------------------------------

    def _grid_distances_to(self, bi: int, bj: int) -> np.ndarray:
        """Return grid distances from every node to the node at ``(bi, bj)``.

        For square topology, slices the precomputed ``_gdist`` table in O(1).
        For hex topology, computes Euclidean distance in Cartesian hex space.

        Parameters
        ----------
        bi, bj : int
            Grid coordinates of the reference node (e.g. the BMU).

        Returns
        -------
        np.ndarray, shape (w, h)
            ``result[i, j]`` = grid distance from node (i, j) to node (bi, bj).

        Notes
        -----
        Grid distance (topological hops) differs from weight-space distance
        (‖W_ij − W_bmu‖). This method returns the former, used to compute
        the Gaussian neighborhood function h(i,j) during weight updates.
        """
        if self.topology == "hex":
            return np.linalg.norm(
                self.hex_coords - self.hex_coords[bi, bj], axis=2  # type: ignore
            )
        return self._gdist[:, :, bi, bj]  # type: ignore[not-subscriptable]

    def get_bmu(self, point: np.ndarray) -> tuple[int, int]:
        """Find the Best Matching Unit (BMU) for a single input vector.

        The BMU is the grid node whose weight vector is closest to ``point``
        under the configured distance metric:

            bmu = argmin_{(i,j)} metric(W_{i,j}, point)

        Parameters
        ----------
        point : np.ndarray, shape (input_dim,)
            A single input sample.

        Returns
        -------
        tuple[int, int]
            ``(i, j)`` grid coordinates of the BMU.

        Example
        -------
        After training a SOM on network traffic features (packet size,
        duration, protocol flags), calling ``get_bmu(x)`` on a new
        connection vector ``x`` returns the grid node that best represents
        that traffic pattern — useful for real-time anomaly scoring.

        Notes
        -----
        For batch inference prefer :meth:`get_bmus`, which vectorizes the
        distance computation across all samples in a single NumPy call.
        """
        dist = self.metric(self.weights, point)
        return np.unravel_index(dist.argmin(), dist.shape)  # type: ignore[invalid-return-type]

    def get_bmus(self, data: np.ndarray) -> np.ndarray:
        """Find the BMU for every sample in ``data`` (vectorized batch search).

        Dispatches to the fastest available implementation:

        1. **Numba JIT** (``use_numba=True``, square + euclidean only):
           parallelized LLVM-compiled kernel; fastest on large grids.
        2. **NumPy broadcast** (euclidean metric):
           ``dist[n,i,j] = ‖W_{i,j} − x_n‖₂``, computed as a single
           einsum-style broadcast; avoids Python loops.
        3. **Generic fallback**: applies ``self.metric`` per sample; supports
           any custom distance function (e.g. cosine, DTW).

        Parameters
        ----------
        data : np.ndarray, shape (n, input_dim)
            Batch of n input samples.

        Returns
        -------
        np.ndarray, shape (n, 2)
            ``result[k]`` = ``[i, j]`` grid coordinates of the BMU for
            sample ``data[k]``.

        Example
        -------
        For a document-topic SOM trained on 300-dim sentence embeddings,
        ``get_bmus(embeddings)`` maps 50,000 documents to their 2D positions
        in a single call (~200ms on CPU for a 30×30 grid), enabling fast
        topic-cluster visualization without per-document Python loops.
        """
        if self.use_numba and euclidean_distances_batch is not None:
            dist = euclidean_distances_batch(
                self.weights.astype(np.float64), data.astype(np.float64)
            )
        elif self.metric is dist_euclidean:
            diff = self.weights[np.newaxis] - data[:, np.newaxis, np.newaxis]
            dist = np.linalg.norm(diff, axis=3)
        else:
            dist = np.stack([self.metric(self.weights, p) for p in data])
        flat = dist.reshape(len(data), -1).argmin(axis=1)
        return np.stack(np.unravel_index(flat, (self.w, self.h)), axis=1)

    # ------------------------------------------------------------------
    # Error metric
    # ------------------------------------------------------------------

    def _calculate_quantization_error(self, data: np.ndarray) -> float:
        """Compute the Mean Quantization Error (MQE) over all training samples.

        The MQE measures how well the prototype vectors represent the data:

            MQE = (1/n) Σᵢ ‖xᵢ − W_{bmu(xᵢ)}‖₂

        Lower is better. A well-trained SOM on structured data typically
        reaches MQE < 0.1 × std(data).

        Implementation note
        -------------------
        Computes the full ``(n, w, h)`` distance matrix once and extracts the
        minimum per sample directly — avoiding the redundant second pass that
        ``get_bmus`` + ``linalg.norm`` would require.

        Interpretation guide
        --------------------
        - **Decreasing MQE**: SOM is actively learning structure.
        - **Plateau early** (< 30% of epochs): map may be under-sized or
          learning rate too low. Consider increasing ``width × height``.
        - **MQE stops decreasing then jumps**: learning rate may be too high,
          causing weight oscillation. Use ``adaptive_lr=True`` in ``fit()``.

        Real-world example
        ------------------
        Training on hourly electricity consumption profiles (24 features per
        day, 365 days): an MQE of 0.05 kWh after 100 epochs means the SOM's
        prototype for each cluster deviates by 50Wh on average from the real
        profiles — acceptable for daily-pattern segmentation.

        Parameters
        ----------
        data : np.ndarray, shape (n, input_dim)
            Dataset to evaluate against current weights.

        Returns
        -------
        float
            Mean L2 distance between each sample and its BMU weight vector.
        """
        if self.use_numba and euclidean_distances_batch is not None:
            # Numba path: returns full (n, w, h) distance matrix
            dist = euclidean_distances_batch(
                self.weights.astype(np.float64), data.astype(np.float64)
            )
        elif self.metric is dist_euclidean:
            # Single broadcast: (n, w, h) — no second pass needed
            diff = self.weights[np.newaxis] - data[:, np.newaxis, np.newaxis]
            dist = np.linalg.norm(diff, axis=3)
        else:
            # Generic fallback for non-euclidean metrics
            dist = np.stack([self.metric(self.weights, p) for p in data])

        # Min over (w, h) — already the BMU distance, no recomputation
        return float(dist.reshape(len(data), -1).min(axis=1).mean())

    # ------------------------------------------------------------------
    # Weight update
    # ------------------------------------------------------------------

    def update_weights(self, data: np.ndarray, lr: float, radius: float) -> None:
        """Update all weights using a vectorized batch-SOM step.

        Algorithm (NumPy path)
        ----------------------
        For each sample xₙ in the batch:

        1. Find BMU: ``bmu_n = argmin_{i,j} dist(W_{i,j}, xₙ)``
        2. Grid distances: ``gd[n, i, j] = d_grid((i,j), bmu_n)``
        3. Gaussian influence (masked before exp):
           ``h[n,i,j] = exp(−gd² / (2σ²))  if gd ≤ 2.5σ,  else 0``
        4. Weighted mean target via einsum (no (n,w,h,d) allocation):
           ``T_{i,j} = Σₙ h[n,i,j]·xₙ / (Σₙ h[n,i,j] + ε)``
        5. Pull weights:
           ``ΔW_{i,j} = η · (T_{i,j} − W_{i,j})``

        The hard cutoff at 2.5σ is applied as a boolean mask *before* the
        exp() call, skipping the expensive computation for ~80% of nodes
        during late training when the radius is small.

        Numba path
        ----------
        When ``use_numba=True``, delegates to ``batch_update_core``, a
        Numba-JIT kernel that performs the same computation but with
        loop-level parallelism and cache-friendly memory access.
        Weights are updated in-place inside the kernel via a pre-cast
        float64 view to avoid silent copy-then-discard bugs.

        Parameters
        ----------
        data : np.ndarray, shape (n, input_dim)
            Shuffled batch of training samples for this epoch.
        lr : float
            Learning rate η ∈ (0, 1]. Controls the magnitude of each
            weight update. Typically decays from ~0.5 to ~0.1 over training.
        radius : float
            Neighbourhood radius σ (in grid units). Controls how broadly
            the BMU's influence propagates. Starts at w/2 and decays to ~1.

        Notes
        -----
        This is a *batch* SOM step, not online (sample-by-sample) SGD.
        Batch updates are more stable and GPU-friendly, but converge slightly
        slower per epoch than online updates on small datasets.

        Example
        -------
        For a 20×20 SOM trained on financial returns (500 assets, 30 features):

        - Epoch 1: ``lr=0.5, radius=10.0`` → wide, coarse reorganization
        - Epoch 25: ``lr=0.35, radius=5.0`` → regional fine-tuning
        - Epoch 50: ``lr=0.20, radius=1.5`` → local convergence

        The radius should reach ~1.0 by the last epoch to ensure each
        node converges to a stable prototype without neighbour interference.
        """
        # --- Numba fast path ---
        if (
            self.use_numba
            and batch_update_core is not None
            and self._gdist is not None
        ):
            # Cast to float64 view in-place — avoids copy-then-discard bug
            # where astype() would return a new array and the kernel mutation
            # would never reach self.weights.
            weights_f64 = self.weights if self.weights.dtype == np.float64 \
                else self.weights.astype(np.float64)
            batch_update_core(
                weights_f64,
                data if data.dtype == np.float64 else data.astype(np.float64),
                lr,
                radius,
                self._gdist if self._gdist.dtype == np.float64 \
                    else self._gdist.astype(np.float64),
            )
            # Write back only if a cast copy was made
            if weights_f64 is not self.weights:
                self.weights[:] = weights_f64
            return

        # --- NumPy fallback ---
        # Step 1: find all BMUs
        if self.metric is dist_euclidean:
            diff   = self.weights[np.newaxis] - data[:, np.newaxis, np.newaxis]
            dist_w = np.linalg.norm(diff, axis=3)  # (n, w, h)
        else:
            dist_w = np.stack([self.metric(self.weights, p) for p in data])
        flat  = dist_w.reshape(len(data), -1).argmin(axis=1)
        bmu_i, bmu_j = np.unravel_index(flat, (self.w, self.h))

        # Step 2: grid distances from every node to each sample's BMU
        if self.topology == "square":
            gd = self._gdist[:, :, bmu_i, bmu_j].transpose(2, 0, 1)  # type: ignore # (n, w, h)
        else:
            gd = np.stack(
                [
                    np.linalg.norm(
                        self.hex_coords - self.hex_coords[bi, bj], axis=2 # type: ignore
                    )
                    for bi, bj in zip(bmu_i, bmu_j)
                ]
            )  # (n, w, h)

        # Step 3: mask BEFORE exp — skips ~80% of nodes in late training
        cutoff = radius * 2.5
        within = gd <= cutoff                              # (n, w, h) bool
        inf    = np.zeros_like(gd)
        inf[within] = np.exp(-(gd[within] ** 2) / (2 * radius ** 2))

        # Step 4: weighted mean target via einsum — avoids (n, w, h, d) alloc
        # 'nij,nd->ijd' contracts the n-axis: Σₙ h[n,i,j] · x[n,d]
        num    = np.einsum("nij,nd->ijd", inf, data, optimize=True)  # (w, h, d)
        denom  = inf.sum(axis=0)[:, :, np.newaxis] + 1e-9            # (w, h, 1)
        target = num / denom                                          # (w, h, d)
        self.weights += lr * (target - self.weights)
        
    # ------------------------------------------------------------------
    # U-Matrix
    # ------------------------------------------------------------------

    def get_u_matrix(self) -> np.ndarray:
        """Compute the Unified Distance Matrix (U-Matrix) for the trained SOM.

        The U-Matrix visualizes cluster boundaries by measuring the average
        weight-space distance between each node and its topological neighbors:

            U[i,j] = (1/|N(i,j)|) Σ_{(k,l) ∈ N(i,j)} ‖W_{i,j} − W_{k,l}‖₂

        High U-Matrix values (bright in a heatmap) indicate a cluster boundary
        — the two neighboring prototypes represent very different inputs.
        Low values (dark) indicate a dense, homogeneous cluster region.

        Returns
        -------
        np.ndarray, shape (w, h)
            U-Matrix values, one per grid node.

        Real-world interpretation
        -------------------------
        **Gene expression SOM** (50×50 grid, ~30k genes):

        - Dark valleys = gene clusters sharing similar expression profiles
          (e.g. immune-response genes, housekeeping genes).
        - Bright ridges = regulatory boundaries between functional groups.
        - Island-shaped dark regions = rare but coherent expression patterns
          (e.g. a tissue-specific gene module).

        **Fraud detection SOM**:

        - Transactions that land near high-U-Matrix nodes are structurally
          ambiguous — they sit between two clusters and warrant closer review.

        Notes
        -----
        Dispatches to :meth:`_u_matrix_square` (vectorized slice pairs) or
        :meth:`_u_matrix_hex` (6-neighbour loop) based on topology.
        """
        if self.topology == "hex":
            return self._u_matrix_hex()
        return self._u_matrix_square()

    def _u_matrix_square(self) -> np.ndarray:
        """Vectorized U-Matrix for square topology using directional slice pairs.

        Iterates over the 4 cardinal directions ``{N, S, E, W}`` and, for each
        direction, computes the weight-space L2 distance between every node
        and its neighbour in that direction using array slicing — no Python
        loops over individual nodes.

        Complexity: O(4 × w × h × d), where d = ``input_dim``.

        Returns
        -------
        np.ndarray, shape (w, h)
            Mean neighbour distance per node.
        """
        u = np.zeros((self.w, self.h))
        count = np.zeros((self.w, self.h))
        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            si = (
                slice(max(0, -di), self.w - max(0, di))
                if di
                else slice(None)
            )
            sj = (
                slice(max(0, -dj), self.h - max(0, dj))
                if dj
                else slice(None)
            )
            ni = (
                slice(max(0, di), self.w - max(0, -di))
                if di
                else slice(None)
            )
            nj = (
                slice(max(0, dj), self.h - max(0, -dj))
                if dj
                else slice(None)
            )
            diff = np.linalg.norm(
                self.weights[si, sj] - self.weights[ni, nj], axis=2
            )
            u[si, sj] += diff
            count[si, sj] += 1
        return u / np.maximum(count, 1)

    def _u_matrix_hex(self) -> np.ndarray:
        """U-Matrix for hexagonal topology using 6-neighbour adjacency.

        Iterates over every node and its valid hex neighbors, accumulating
        L2 weight-space distances. Border nodes have fewer than 6 neighbors
        and are normalized accordingly (no padding artefact).

        Complexity: O(6 × w × h × d) in the worst case (interior nodes).

        Returns
        -------
        np.ndarray, shape (w, h)
            Mean 6-neighbour distance per node.

        Notes
        -----
        Uses :meth:`_get_hex_neighbors` for neighbour index generation with
        boundary clamping, ensuring consistent normalization at grid edges.
        """
        u = np.zeros((self.w, self.h))
        count = np.zeros((self.w, self.h))
        for i in range(self.w):
            for j in range(self.h):
                for ni, nj in self._get_hex_neighbors(i, j):
                    u[i, j] += np.linalg.norm(self.weights[i, j] - self.weights[ni, nj])
                    count[i, j] += 1
        return u / np.maximum(count, 1)

    def _get_hex_neighbors(self, i: int, j: int) -> list[tuple[int, int]]:
        """Return valid 6-neighbour indices for hex node ``(i, j)``.

        In a hex offset grid the 6 neighbors of ``(i, j)`` differ by column
        parity:

        Even column (j % 2 == 0)::

            (i-1, j)  (i+1, j)
            (i,   j-1) (i,  j+1)
            (i-1, j-1) (i-1, j+1)   ← left-leaning diagonals

        Odd column (j % 2 == 1)::

            (i-1, j)  (i+1, j)
            (i,   j-1) (i,  j+1)
            (i+1, j-1) (i+1, j+1)   ← right-leaning diagonals

        Parameters
        ----------
        i, j : int
            Grid coordinates of the query node.

        Returns
        -------
        list[tuple[int, int]]
            Valid neighbour indices within the ``[0, w) × [0, h)`` grid bounds.
        """
        if j % 2 == 0:
            cands = [
                (i - 1, j), (i + 1, j),
                (i, j - 1), (i, j + 1),
                (i - 1, j - 1), (i + 1, j - 1),
            ]
        else:
            cands = [
                (i - 1, j), (i + 1, j),
                (i, j - 1), (i, j + 1),
                (i - 1, j + 1), (i + 1, j + 1),
            ]
        return [
            (ni, nj)
            for ni, nj in cands
            if 0 <= ni < self.w and 0 <= nj < self.h
        ]

    # ------------------------------------------------------------------
    # Activation map
    # ------------------------------------------------------------------

    def get_activation_map(self, bmus: np.ndarray) -> np.ndarray:
        """Count BMU hits per grid node from a precomputed *bmus* array.

        Also known as a **hit histogram** or **response map**. Each cell
        ``heatmap[i, j]`` records how many input samples selected node (i, j)
        as their BMU.

        Uses ``np.add.at`` for correct unbuffered in-place accumulation,
        handling repeated indices without summation artefacts.

        Parameters
        ----------
        bmus : np.ndarray, shape (n, 2)
            Precomputed BMU indices from :meth:`get_bmus`. Reuse the same
            array computed during or after training to avoid redundant
            distance calculations.

        Returns
        -------
        np.ndarray, shape (w, h), dtype int
            Hit count per grid node.

        Interpretation
        --------------
        - **Uniform map**: hits spread evenly → good data coverage, no dead nodes.
        - **Dead nodes** (count = 0): weight vector never won any BMU contest.
          Symptom of over-sized grid or poor initialization.
          Fix: reduce grid size, use PCA init, or train longer.
        - **Hot spots** (very high count): dominant cluster in the data.
          Useful for density estimation without kernel smoothing.

        Example
        -------
        Training a 15×15 SOM on 3 years of hourly energy consumption data
        (26,280 samples, 24 features per day):

        - The activation map reveals which daily-consumption profiles are
          most common (e.g. weekday-peak nodes with 1,000+ hits vs.
          holiday-flat-load nodes with ~50 hits).
        - Dead nodes at map edges often correspond to extreme outliers
          (storm days, grid outages) and can be flagged for anomaly review.
        """
        heatmap = np.zeros((self.w, self.h), dtype=int)
        np.add.at(heatmap, (bmus[:, 0], bmus[:, 1]), 1)
        return heatmap

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        data: np.ndarray,
        epochs: int,
        learning_rate: float,
        snapshot_every: int = 4,
        adaptive_lr: bool = False,
        decay_schedule: Decay_Schedule | None = None,
    ) -> list[SOM]:
        """Train the SOM and return periodic weight snapshots.

        Training loop
        -------------
        For each epoch ``ep`` in ``[0, epochs)``:

        1. **Schedule**: compute decaying lr and radius using the active
           decay schedule (``'linear'`` or ``'exponential'``):

           *Linear* (default):

           .. code-block::

               progress  = ep / epochs
               lr        = η₀ · (1 − progress · decay_factor)
               radius    = max(1.0, (w/2) · (1 − progress · 0.85))

           *Exponential* (Kohonen 1995 canonical):

           .. code-block::

               τ  = epochs / 3
               lr     = η₀ · exp(−ep / τ)
               radius = max(1.0, (w/2) · exp(−ep / τ))

        2. **Shuffle**: permute sample order to reduce batch correlation.
        3. **Update**: call :meth:`update_weights` with the full shuffled dataset.
        4. **Record**: compute MQE and append to ``error_history_``.
        5. **Snapshot**: deep-copy weights every ``snapshot_every`` epochs
           (and always on the last epoch).

        Adaptive learning rate
        ----------------------
        When ``adaptive_lr=True``, the decay factor adjusts based on the
        derivative of the quantization error ΔQE = QE_ep − QE_{ep-1}:

        - ``ΔQE < −ε`` (steep improvement): slow down decay
          (``decay_factor *= 0.95``) to preserve momentum.
        - ``|ΔQE| < ε`` (plateau): speed up decay
          (``decay_factor *= 1.05``) to escape the flat region.

        For the exponential schedule, ``adaptive_lr`` modulates ``τ`` instead
        of ``decay_factor``: a steep drop stretches τ (slower decay), a plateau
        compresses it (faster decay).

        This is analogous to ReduceLROnPlateau in deep learning, adapted
        to the SOM's non-gradient training dynamic.

        Parameters
        ----------
        data : np.ndarray, shape (n, input_dim)
            Full training dataset. Shuffled internally each epoch.
        epochs : int
            Number of training epochs. Rule of thumb: ``epochs ≈ 500 × (w × h) / n``.
            For a 20×20 SOM on 10,000 samples → ~20 epochs minimum;
            100 epochs for stable convergence.
        learning_rate : float
            Initial learning rate η₀. Typical range: 0.3–0.8.

            - Too high (> 0.9): weights oscillate, MQE never plateaus.
            - Too low (< 0.05): map takes many epochs to organize.
            - **Recommended**: start at 0.5, tune via ``error_history_``.

        snapshot_every : int, default 4
            Save a weight snapshot every N epochs. Lower values → more
            animation frames (useful for visualization); higher values →
            less memory usage. Set to ``epochs`` to keep only the final state.
        adaptive_lr : bool, default False
            Enable derivative-based learning rate scheduling.
            Recommended for long training runs (> 100 epochs) or when
            MQE curves show early plateaus or oscillation.
        decay_schedule : Decay_Schedule | None, default ``None``
            Override the instance-level decay schedule for this training run.
            ``None`` falls back to ``self.decay_schedule`` set at construction.

            - ``'linear'``: η decays as ``η₀ · (1 − t/T · f)``.
              Simpler to reason about; ``decay_factor`` is directly
              interpretable as the fraction of η₀ consumed by epoch T.
            - ``'exponential'``: η decays as ``η₀ · exp(−t/τ)``,
              τ = T/3. Kohonen's original formulation; steeper early
              decay, gentler tail — often better for large grids.

        Returns
        -------
        list[SOM]
            Ordered list of SOM weight snapshots. Each element is a
            shallow-copied SOM with ``weights`` deep-copied at that epoch.
            The last element always corresponds to the final trained state
            (same object as ``self`` in terms of weights).

        Example — anomaly detection on network traffic
        -----------------------------------------------
        Train a 25×25 SOM on 100,000 connection records (20 features):

        >>> snapshots = som.fit(
        ...     data,
        ...     epochs=100,
        ...     learning_rate=0.5,
        ...     snapshot_every=10,
        ...     adaptive_lr=True,
        ... )
        >>> import matplotlib.pyplot as plt
        >>> plt.plot(som.error_history_)
        >>> plt.title("MQE convergence — should plateau by epoch 60")

        After training:

        - Records mapping to high-U-Matrix nodes = structurally ambiguous
          (possible port-scan or DDoS traffic).
        - Records with MQE contribution > 2σ above mean = true anomalies
          (novel attack patterns not covered by any prototype).

        Notes
        -----
        The snapshot list can be used to animate the SOM's self-organization
        process — particularly useful for presentations and debugging maps
        that fail to converge (e.g. grid too large, poor metric choice).
        """
        snapshots: list[SOM] = []
        rng_epoch = np.random.default_rng(99)
        self.error_history_ = []

        # Resolve schedule: fit() arg overrides __init__ default
        schedule = decay_schedule if decay_schedule is not None else self.decay_schedule

        # Linear schedule state — lr decays to lr0 * (1 - decay_factor) over training
        decay_factor = 0.8

        # Exponential schedule state — τ = T/3 gives η(T) ≈ 0.05·η₀,
        # consistent with linear's end value at decay_factor=0.8
        tau = epochs / 3.0

        prev_qe: float | None = None

        for ep in range(epochs):
            progress = ep / epochs

            if schedule == "exponential":
                lr     = learning_rate * np.exp(-ep / tau)
                radius = max(1.0, (self.w / 2) * np.exp(-ep / tau))
            else:
                lr     = learning_rate * (1 - progress * decay_factor)
                radius = max(1.0, (self.w / 2) * (1 - progress * 0.85))

            idx = rng_epoch.permutation(len(data))
            self.update_weights(data[idx], lr, radius)

            # --- Compute quantization error ---
            qe = self._calculate_quantization_error(data)
            self.error_history_.append(qe)

            # --- Adaptive lr scheduling (derivative of QE) ---
            if adaptive_lr and prev_qe is not None:
                dqe = qe - prev_qe  # negative = improvement
                if dqe < -1e-6:
                    # Steep improvement → slow down decay
                    if schedule == "exponential":
                        tau = min(epochs, tau / 0.95)       # stretch τ
                    else:
                        decay_factor = max(0.4, decay_factor * 0.95)
                elif abs(dqe) < 1e-6:
                    # Plateau → speed up decay
                    if schedule == "exponential":
                        tau = max(epochs / 10.0, tau * 0.95)  # compress τ
                    else:
                        decay_factor = min(0.95, decay_factor * 1.05)
            prev_qe = qe

            if ep % snapshot_every == 0 or ep == epochs - 1:
                snap = SOM.__new__(SOM)
                snap.w, snap.h       = self.w, self.h
                snap.input_dim       = self.input_dim
                snap.topology        = self.topology
                snap.metric          = self.metric
                snap.weights         = self.weights.copy()
                snap.hex_coords      = self.hex_coords
                snap._gdist          = self._gdist  # immutable, safe to share
                snap.use_numba       = self.use_numba
                snap.decay_schedule  = self.decay_schedule
                snap.error_history_  = list(self.error_history_)
                snapshots.append(snap)

        return snapshots