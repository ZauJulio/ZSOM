"""
Error metrics for Self-Organizing Map quality assessment.

Provides two categories of metrics:

**Per-sample metrics** — measure the distance between a single input
vector and its BMU weight vector:

- :func:`qE` — Quantization Error (L2 / Euclidean distance)
- :func:`MAE` — Mean Absolute Error
- :func:`MSE` — Mean Squared Error
- :func:`RMSE` — Root Mean Squared Error
- :func:`MAPE` — Mean Absolute Percentage Error

**Per-node aggregate metrics** — compute the mean error for all samples
assigned to each SOM node, returning a ``(w, h)`` grid:

- :func:`qE_node` — Mean quantization error per node
- :func:`MAE_node` — Mean MAE per node
- :func:`MSE_node` — Mean MSE per node
- :func:`RMSE_node` — Mean RMSE per node
- :func:`MAPE_node` — Mean MAPE per node

All per-sample metrics are fully vectorized with NumPy.  Per-node metrics
use the SOM's ``get_bmus`` to group samples by node, then compute the
aggregate efficiently.

Usage::

    from zsom import SOM, make_clusters
    from zsom.error_metrics import qE_node, RMSE_node
    import numpy as np

    rng = np.random.default_rng(42)
    pts, labels = make_clusters(300, rng)
    som = SOM(12, 12, 2, rng)
    som.fit(pts, epochs=80, learning_rate=0.5)

    bmus = som.get_bmus(pts)
    qe_grid = qE_node(som, pts, bmus)
    rmse_grid = RMSE_node(som, pts, bmus)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .som import SOM


# ---------------------------------------------------------------------------
# Per-sample metrics
# ---------------------------------------------------------------------------

def qE(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Quantization Error — Euclidean (L2) distance between vectors.

    Formula::

        qE(a, b) = √Σ(aᵢ − bᵢ)²

    This is the standard SOM quality metric.  Lower values indicate that
    the BMU weight vector is a better representative of its assigned input.
    It measures the *geometric* distance between input and prototype.

    **When to use**: Always — qE is the primary quality metric for SOMs.
    It tells you how well the weight vectors approximate their Voronoi
    regions.

    Args:
        a: Array of shape ``(d,)`` or ``(n, d)`` — input vector(s).
        b: Array of shape ``(d,)`` or ``(n, d)`` — BMU weight vector(s).

    Returns:
        Scalar or array of Euclidean distances.
    """
    diff = np.subtract(a, b)
    return np.sqrt(np.sum(np.square(diff), axis=-1))


def MAE(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Mean Absolute Error — average of |aᵢ − bᵢ| across dimensions.

    Formula::

        MAE(a, b) = (1/d) Σ|aᵢ − bᵢ|

    More robust to outliers than MSE because errors are not squared.
    Provides a linear, interpretable error in the original units.

    **When to use**: When you want an error metric that treats all
    dimensional deviations equally and is not amplified by outliers.

    Args:
        a: Array of shape ``(d,)`` or ``(n, d)``.
        b: Array of shape ``(d,)`` or ``(n, d)``.

    Returns:
        Scalar or array of mean absolute errors.
    """
    return np.abs(np.subtract(a, b)).mean(axis=-1)


def MSE(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Mean Squared Error — average of (aᵢ − bᵢ)² across dimensions.

    Formula::

        MSE(a, b) = (1/d) Σ(aᵢ − bᵢ)²

    Penalizes large deviations more than small ones due to squaring.
    Often used as a loss function in machine learning.

    **When to use**: When you want to penalize large per-dimension
    deviations disproportionately, or need a differentiable loss.

    Args:
        a: Array of shape ``(d,)`` or ``(n, d)``.
        b: Array of shape ``(d,)`` or ``(n, d)``.

    Returns:
        Scalar or array of mean squared errors.
    """
    return np.square(np.subtract(a, b)).mean(axis=-1)


def RMSE(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Root Mean Squared Error — √MSE.

    Formula::

        RMSE(a, b) = √((1/d) Σ(aᵢ − bᵢ)²)

    Has the same units as the original data, making it more interpretable
    than MSE, while still penalizing large deviations.

    **When to use**: When you want the interpretability of MAE but the
    outlier-sensitivity of MSE.  RMSE ≥ MAE always, with equality only
    when all per-dimension errors are identical.

    Args:
        a: Array of shape ``(d,)`` or ``(n, d)``.
        b: Array of shape ``(d,)`` or ``(n, d)``.

    Returns:
        Scalar or array of root mean squared errors.
    """
    return np.sqrt(np.square(np.subtract(a, b)).mean(axis=-1))


def MAPE(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Mean Absolute Percentage Error — relative error in percent.

    Formula::

        MAPE(a, b) = (100/d) Σ|((aᵢ − bᵢ) / aᵢ)|

    Expresses the error as a percentage of the true value.  Undefined
    (infinite) when any ``aᵢ = 0``; a small epsilon is added for
    numerical stability.

    **When to use**: When you need a scale-independent quality metric
    and the true values are strictly positive (e.g. prices, counts,
    physical measurements).

    **Caveat**: Not suitable for data with zero-valued features.

    Args:
        a: Array of shape ``(d,)`` or ``(n, d)`` — true values.
        b: Array of shape ``(d,)`` or ``(n, d)`` — predicted / BMU values.

    Returns:
        Scalar or array of MAPE values (in percent).
    """
    return np.abs(np.divide(np.subtract(a, b), a + 1e-9)).mean(axis=-1) * 100


# ---------------------------------------------------------------------------
# Per-node aggregate metrics
# ---------------------------------------------------------------------------

def _per_node_metric(
    som: SOM,
    pts: np.ndarray,
    bmus: np.ndarray,
    metric_fn,
) -> np.ndarray:
    """Compute a per-node aggregate metric.

    Groups samples by their BMU, computes the sample-level metric for each
    sample against its BMU weight, then averages within each node.

    Args:
        som:       Trained SOM instance.
        pts:       Input array of shape ``(n, input_dim)``.
        bmus:      Precomputed BMU indices ``(n, 2)``.
        metric_fn: One of the per-sample metric functions.

    Returns:
        Float array of shape ``(w, h)`` with mean metric per node.
        Nodes with no assigned samples get value 0.
    """
    result = np.zeros((som.w, som.h))
    counts = np.zeros((som.w, som.h))

    # Batch: compute metric for all samples at once
    bmu_weights = som.weights[bmus[:, 0], bmus[:, 1]]  # (n, d)
    errors = metric_fn(pts, bmu_weights)  # (n,)

    # Accumulate per node
    np.add.at(result, (bmus[:, 0], bmus[:, 1]), errors)
    np.add.at(counts, (bmus[:, 0], bmus[:, 1]), 1)

    # Average (avoid division by zero)
    mask = counts > 0
    result[mask] /= counts[mask]
    result[~mask] = np.nan
    return result


def qE_node(
    som: SOM,
    pts: np.ndarray,
    bmus: np.ndarray,
) -> np.ndarray:
    """Mean quantization error per SOM node.

    For each node, computes the average Euclidean distance between the
    node's weight vector and all input samples assigned to it.

    Lower values indicate better local representation quality.

    Args:
        som:  Trained SOM instance.
        pts:  Input array of shape ``(n, input_dim)``.
        bmus: Precomputed BMU indices ``(n, 2)``.

    Returns:
        Float array of shape ``(w, h)``.
    """
    return _per_node_metric(som, pts, bmus, qE)


def MAE_node(
    som: SOM,
    pts: np.ndarray,
    bmus: np.ndarray,
) -> np.ndarray:
    """Mean MAE per SOM node.

    For each node, computes the average Mean Absolute Error between the
    node's weight vector and all input samples assigned to it.

    Args:
        som:  Trained SOM instance.
        pts:  Input array of shape ``(n, input_dim)``.
        bmus: Precomputed BMU indices ``(n, 2)``.

    Returns:
        Float array of shape ``(w, h)``.
    """
    return _per_node_metric(som, pts, bmus, MAE)


def MSE_node(
    som: SOM,
    pts: np.ndarray,
    bmus: np.ndarray,
) -> np.ndarray:
    """Mean MSE per SOM node.

    For each node, computes the average Mean Squared Error between the
    node's weight vector and all input samples assigned to it.

    Args:
        som:  Trained SOM instance.
        pts:  Input array of shape ``(n, input_dim)``.
        bmus: Precomputed BMU indices ``(n, 2)``.

    Returns:
        Float array of shape ``(w, h)``.
    """
    return _per_node_metric(som, pts, bmus, MSE)


def RMSE_node(
    som: SOM,
    pts: np.ndarray,
    bmus: np.ndarray,
) -> np.ndarray:
    """Mean RMSE per SOM node.

    For each node, computes the average Root Mean Squared Error between
    the node's weight vector and all input samples assigned to it.

    Args:
        som:  Trained SOM instance.
        pts:  Input array of shape ``(n, input_dim)``.
        bmus: Precomputed BMU indices ``(n, 2)``.

    Returns:
        Float array of shape ``(w, h)``.
    """
    return _per_node_metric(som, pts, bmus, RMSE)


def MAPE_node(
    som: SOM,
    pts: np.ndarray,
    bmus: np.ndarray,
) -> np.ndarray:
    """Mean MAPE per SOM node.

    For each node, computes the average Mean Absolute Percentage Error
    between the node's weight vector and all input samples assigned to it.

    **Caveat**: MAPE is not suitable for data with zero-valued features.

    Args:
        som:  Trained SOM instance.
        pts:  Input array of shape ``(n, input_dim)``.
        bmus: Precomputed BMU indices ``(n, 2)``.

    Returns:
        Float array of shape ``(w, h)`` with MAPE values in percent.
    """
    return _per_node_metric(som, pts, bmus, MAPE)
