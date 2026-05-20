"""
Distance metrics for the Self-Organizing Map.

Each metric function has the signature::

    (weights: ndarray[w, h, d], pt: ndarray[d]) -> ndarray[w, h]

Available metrics
-----------------
euclidean
    L2 norm — the default SOM metric. Gives equal weight to all dimensions
    and produces isotropic (circular) Voronoi cells.  Best for continuous
    data where all features share a comparable scale (e.g. sensor readings,
    spatial coordinates).  Sensitive to outliers because deviations are
    squared before summing.

manhattan
    L1 norm — sum of absolute differences.  More robust to outliers than
    L2 because large single-dimension deviations are not amplified.  Best
    for sparse or high-dimensional tabular data.  Produces diamond-shaped
    Voronoi cells.

chebyshev
    L-infinity norm — the maximum absolute coordinate difference.  Only the
    single worst-case dimension matters.  Best for problems where uniform
    tolerance across all dimensions is required (e.g. grid-based game
    movement, QA checks).  Equivalent to Minkowski as p → ∞.  Produces
    square-aligned Voronoi cells.

cosine
    ``1 - cosine_similarity`` — an angle-based metric that is completely
    invariant to vector magnitude.  Best for data where *direction* matters
    more than *scale* — e.g. TF-IDF document vectors, word embeddings,
    spectral signatures.  Range: 0 (identical direction) to 2 (opposite).

minkowski
    ``L-p`` norm — a generalization that smoothly interpolates between
    L1 (p=1), L2 (p=2), and L∞ (p→∞).  Non-integer *p* values are valid
    and provide a custom trade-off between outlier robustness (low p) and
    sensitivity to the dominant dimension (high p).  Use when you need to
    fine-tune the balance between L1 and L2 behavior.
"""

from __future__ import annotations

from typing import Callable, Literal

import numpy as np

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

MetricName = Literal["euclidean", "manhattan", "chebyshev", "cosine", "minkowski"]
"""Valid metric names for the SOM distance function."""

DistanceFunc = Callable[[np.ndarray, np.ndarray], np.ndarray]
"""Signature of a distance function: ``(weights[w,h,d], pt[d]) -> dist[w,h]``."""


# ---------------------------------------------------------------------------
# Distance functions
# ---------------------------------------------------------------------------

def dist_euclidean(weights: np.ndarray, pt: np.ndarray) -> np.ndarray:
    """Euclidean (L2) distance from every weight vector to *pt*.

    Formula::

        d(w, p) = √Σ(wᵢ − pᵢ)²

    The standard SOM distance metric. Gives equal weight to all dimensions
    and produces isotropic Voronoi cells in weight space.

    **Best for**: Continuous, real-valued data where all features are on a
    comparable scale — e.g. sensor measurements, spatial coordinates,
    normalized embeddings.

    **Caveat**: Sensitive to outliers and to features with very different
    scales (standardize first).

    Args:
        weights: Array of shape ``(w, h, d)`` with node weight vectors.
        pt:      Input point of shape ``(d,)``.

    Returns:
        Float array of shape ``(w, h)`` with L2 distance from each node to *pt*.
    """
    return np.linalg.norm(weights - pt, axis=2)


def dist_manhattan(weights: np.ndarray, pt: np.ndarray) -> np.ndarray:
    """Manhattan (L1) distance from every weight vector to *pt*.

    Formula::

        d(w, p) = Σ|wᵢ − pᵢ|

    Sum of absolute coordinate differences. More robust to outliers than
    L2 because large deviations in a single dimension are not squared.
    Produces diamond-shaped Voronoi cells.

    **Best for**: High-dimensional tabular data, sparse features, or data
    with occasional outliers — e.g. customer behavior vectors, survey
    responses, categorical-encoded features.

    **Caveat**: May under-weight correlated dimensions.

    Args:
        weights: Array of shape ``(w, h, d)``.
        pt:      Input point of shape ``(d,)``.

    Returns:
        Float array of shape ``(w, h)`` with L1 distance from each node to *pt*.
    """
    return np.abs(weights - pt).sum(axis=2)


def dist_chebyshev(weights: np.ndarray, pt: np.ndarray) -> np.ndarray:
    """Chebyshev (L-infinity) distance from every weight vector to *pt*.

    Formula::

        d(w, p) = max|wᵢ − pᵢ|

    Maximum absolute coordinate difference across all dimensions. Sensitive
    to the single dimension with the largest deviation; produces
    square-aligned Voronoi cells. Equivalent to L-p as p → ∞.

    **Best for**: Grid-based movement (chess-king distance), uniform
    tolerance QA checks, or any scenario where the worst-case dimension
    is the bottleneck.

    **Caveat**: Ignores contributions from all dimensions except the
    most deviant.

    Args:
        weights: Array of shape ``(w, h, d)``.
        pt:      Input point of shape ``(d,)``.

    Returns:
        Float array of shape ``(w, h)`` with L-inf distance from each node to *pt*.
    """
    return np.abs(weights - pt).max(axis=2)


def dist_cosine(weights: np.ndarray, pt: np.ndarray) -> np.ndarray:
    """Cosine dissimilarity ``1 - cosine_similarity`` from weights to *pt*.

    Formula::

        d(w, p) = 1 − (w · p) / (‖w‖ · ‖p‖)

    Angle-based metric; invariant to the magnitude of weight vectors and
    input. Best suited for normalized or high-dimensional data (e.g. word
    embeddings, TF-IDF vectors) where direction matters more than scale.
    Zero means identical direction; 2.0 means opposite directions.

    **Best for**: Text/NLP embeddings, spectral signatures, any data where
    you care about the *shape* of the vector rather than its *length*.

    **Caveat**: Undefined when either vector is zero; handled here via a
    small epsilon for numerical stability.

    Args:
        weights: Array of shape ``(w, h, d)``.
        pt:      Input point of shape ``(d,)``.

    Returns:
        Float array of shape ``(w, h)`` with cosine dissimilarity.
    """
    wn = weights / (np.linalg.norm(weights, axis=2, keepdims=True) + 1e-9)
    pn = pt / (np.linalg.norm(pt) + 1e-9)
    return 1.0 - (wn * pn).sum(axis=2)


def dist_minkowski(weights: np.ndarray, pt: np.ndarray, p: float = 3.0) -> np.ndarray:
    """Minkowski L-p distance from every weight vector to *pt*.

    Formula::

        d(w, p) = (Σ|wᵢ − pᵢ|^p)^(1/p)

    The Minkowski family unifies L1 (p=1), L2 (p=2), and L-inf (p→∞).
    Non-integer *p* values are valid and interpolate between the standard
    norms. Larger *p* values increasingly focus on the dominant-deviation
    dimension.

    **Best for**: Fine-tuning the trade-off between robustness (low p, close
    to L1) and sensitivity to outliers / dominant dimensions (high p, close
    to L∞).  A common choice is p=3 as a compromise.

    Args:
        weights: Array of shape ``(w, h, d)``.
        pt:      Input point of shape ``(d,)``.
        p:       Minkowski exponent. Must be >= 1. Defaults to 3.0.
                 To use a different `p`, pass a lambda, e.g.,
                 ``metric=lambda w, x: dist_minkowski(w, x, p=4)``.

    Returns:
        Float array of shape ``(w, h)`` with Minkowski distance.

    Raises:
        ValueError: If ``p < 1`` (would violate the triangle inequality).
    """
    if p < 1:
        raise ValueError(f"Minkowski p must be >= 1, got {p}")
    return (np.abs(weights - pt) ** p).sum(axis=2) ** (1.0 / p)


METRICS: dict[MetricName, DistanceFunc] = {
    "euclidean": dist_euclidean,
    "manhattan": dist_manhattan,
    "chebyshev": dist_chebyshev,
    "cosine": dist_cosine,
    "minkowski": dist_minkowski,
}
"""Registry of built-in distance functions, keyed by canonical name."""
