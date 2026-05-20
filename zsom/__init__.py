"""
ZSOM — Self-Organizing Map in pure NumPy with optional Numba acceleration.

A SOM is an unsupervised competitive neural network that projects
high-dimensional input data onto a low-dimensional (2D) discrete grid
while preserving the topological structure of the input space.

Quick start::

    from zsom import SOM, make_clusters
    import numpy as np

    rng = np.random.default_rng(42)
    pts, labels = make_clusters(300, rng)
    som = SOM(12, 12, 2, rng)
    snapshots = som.fit(pts, epochs=80, lr0=0.5)
"""

from __future__ import annotations

__version__ = "1.0.0"

# Core
from .som import SOM, hex_grid_coords, Topology, InitMethod

# Metrics
from .metrics import (
    METRICS,
    dist_chebyshev,
    dist_cosine,
    dist_euclidean,
    dist_manhattan,
    dist_minkowski,
    MetricName,
    DistanceFunc,
)

# Numba
from .numba_accel import HAS_NUMBA

# Error metrics
from .error_metrics import (
    qE,
    MAE,
    MSE,
    RMSE,
    MAPE,
    qE_node,
    MAE_node,
    MSE_node,
    RMSE_node,
    MAPE_node,
)



__all__ = [
    "__version__",
    # Core
    "SOM",
    "hex_grid_coords",
    # Type aliases
    "Topology",
    "InitMethod",
    "MetricName",
    "DistanceFunc",
    # Metrics
    "METRICS",
    "dist_euclidean",
    "dist_manhattan",
    "dist_chebyshev",
    "dist_cosine",
    "dist_minkowski",
    # Numba
    "HAS_NUMBA",
    # Error metrics
    "qE",
    "MAE",
    "MSE",
    "RMSE",
    "MAPE",
    "qE_node",
    "MAE_node",
    "MSE_node",
    "RMSE_node",
    "MAPE_node",
]
