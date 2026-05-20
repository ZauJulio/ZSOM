#!/usr/bin/env python3
"""Quick demo: train a SOM on three Gaussian clusters and animate.

Usage::

    python examples/demo_clusters.py
    python examples/demo_clusters.py --static
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from zsom import SOM
from examples.datasets import make_clusters
from examples.visualization import plot_animated, plot_static

GRID = 12
EPOCHS = 80
LR = 0.5
N_POINTS = 300
SEED = 42
SNAPSHOT_EVERY = 4


def run(static: bool = False) -> None:
    rng = np.random.default_rng(SEED)
    pts, labels = make_clusters(N_POINTS, rng)

    som = SOM(GRID, GRID, pts.shape[1], rng)
    print(f"Training {GRID}×{GRID} SOM on {N_POINTS} cluster points…")
    snapshots = som.fit(pts, EPOCHS, LR, snapshot_every=SNAPSHOT_EVERY)
    print("Done.")

    if static:
        plot_static(som, pts, labels, dataset_name="clusters")
    else:
        plot_animated(
            snapshots, pts, labels, EPOCHS, SNAPSHOT_EVERY,
            dataset_name="clusters",
        )


if __name__ == "__main__":
    import sys
    run(static="--static" in sys.argv)
