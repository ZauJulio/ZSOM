"""
Optional Numba-accelerated kernels for hot SOM operations.

If `numba` is installed, this module exposes JIT-compiled versions of the
two most expensive SOM routines:

- :func:`euclidean_distances_batch` — compute ``(n, w, h)`` distance matrix
- :func:`batch_update_core` — fused BMU search + Gaussian neighborhood update

When Numba is **not** available the module-level flag :data:`HAS_NUMBA` is
``False`` and the accelerated functions are set to ``None``.  The SOM class
checks this flag and falls back to pure NumPy automatically.

Usage::

    from zsom.numba_accel import HAS_NUMBA
    print(f"Numba acceleration: {'enabled' if HAS_NUMBA else 'disabled'}")
"""

from __future__ import annotations

import numpy as np

HAS_NUMBA: bool = False
euclidean_distances_batch = None
batch_update_core = None

try:
    from numba import njit, prange  # type: ignore[import-untyped]

    HAS_NUMBA = True
    raise ImportError("Numba not installed")

    @njit(cache=True, parallel=True)
    def _euclidean_distances_batch(
        weights: np.ndarray, pts: np.ndarray
    ) -> np.ndarray:
        """Compute L2 distances from every point to every weight node.

        Args:
            weights: ``(w, h, d)`` weight array.
            pts:     ``(n, d)`` input points.

        Returns:
            ``(n, w, h)`` distance matrix.
        """
        n = pts.shape[0]
        w = weights.shape[0]
        h = weights.shape[1]
        d = weights.shape[2]
        out = np.empty((n, w, h), dtype=np.float64)
        for k in prange(n):  # ty:ignore[not-iterable]
            for i in range(w):
                for j in range(h):
                    s = 0.0
                    for dd in range(d):
                        diff = weights[i, j, dd] - pts[k, dd]
                        s += diff * diff
                    out[k, i, j] = np.sqrt(s)
        return out

    euclidean_distances_batch = _euclidean_distances_batch

    @njit(cache=True, parallel=True)
    def _batch_update_core(
        weights: np.ndarray,
        pts: np.ndarray,
        lr: float,
        radius: float,
        gdist: np.ndarray,
    ) -> None:
        """Fused BMU search + Gaussian neighborhood weight update.

        Modifies *weights* in-place.  For each sample, finds the BMU via
        exhaustive L2 search, then pulls all nodes within ``2.5 × radius``
        toward the sample.

        Args:
            weights: ``(w, h, d)`` — modified in place.
            pts:     ``(n, d)`` input points.
            lr:      Learning rate for this step.
            radius:  Gaussian neighborhood σ.
            gdist:   ``(w, h, w, h)`` precomputed grid-distance table.
        """
        n = pts.shape[0]
        w = weights.shape[0]
        h = weights.shape[1]
        d = weights.shape[2]
        cutoff = radius * 2.5
        two_sigma_sq = 2.0 * radius * radius

        # --- accumulate numerator / denominator per node ---
        num = np.zeros((w, h, d), dtype=np.float64)
        denom = np.zeros((w, h), dtype=np.float64)

        for k in range(n):
            # find BMU for sample k
            best_dist = np.inf
            bi, bj = 0, 0
            for i in range(w):
                for j in range(h):
                    s = 0.0
                    for dd in range(d):
                        diff = weights[i, j, dd] - pts[k, dd]
                        s += diff * diff
                    if s < best_dist:
                        best_dist = s
                        bi, bj = i, j

            # accumulate weighted influence
            for i in prange(w):  # ty:ignore[not-iterable]
                for j in range(h):
                    gd = gdist[i, j, bi, bj]
                    if gd <= cutoff:
                        inf = np.exp(-(gd * gd) / two_sigma_sq)
                        denom[i, j] += inf
                        for dd in range(d):
                            num[i, j, dd] += inf * pts[k, dd]

        # --- pull weights toward weighted mean ---
        for i in prange(w):  # ty:ignore[not-iterable]
            for j in range(h):
                den = denom[i, j] + 1e-9
                for dd in range(d):
                    target = num[i, j, dd] / den
                    weights[i, j, dd] += lr * (target - weights[i, j, dd])

    batch_update_core = _batch_update_core

except ImportError:
    pass
