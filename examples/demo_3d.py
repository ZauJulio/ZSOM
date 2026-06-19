#!/usr/bin/env python3
"""Demo: train a SOM on a 3D object and animate.

Usage::

    python examples/demo_3d.py
    python examples/demo_3d.py --static
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from zsom import SOM
from examples.datasets import ObjectName, make_obj
from examples.visualization import plot_animated, plot_static

GRID = 32
EPOCHS = 1000
LR = 0.6
N_POINTS = 600
SEED = 42
SNAPSHOT_EVERY = 4
TOPOLOGY = "hex"
DECAY_SCHEDULE = "exponential"
INIT = "pca"

def run(static: bool = False, object_name: ObjectName = "vader") -> None:
    rng = np.random.default_rng(SEED)
    pts, labels = make_obj(N_POINTS, rng, obj=object_name)

    som = SOM(GRID, GRID, pts.shape[1], rng, init=INIT, data=pts, topology=TOPOLOGY, decay_schedule=DECAY_SCHEDULE)
    print(f"Training {GRID}×{GRID} SOM on {len(pts)} {object_name} points (3D)…")
    snapshots = som.fit(pts, EPOCHS, LR, snapshot_every=SNAPSHOT_EVERY)
    print("Done.")

    if static:
        plot_static(som, pts, labels, dataset_name=object_name)
    else:
        plot_animated(
            snapshots, pts, labels, EPOCHS, SNAPSHOT_EVERY,
            dataset_name=object_name,
        )
        
        import matplotlib.pyplot as plt
        
        plt.plot(som.error_history_)
        plt.xlabel("Epoch")
        plt.ylabel("MQE")
        plt.title("SOM convergence")
        plt.show()


if __name__ == "__main__":
    static = "--static" in sys.argv
    object_name: ObjectName = "vader"
    
    for arg in sys.argv[1:]:
        if arg == "--static":
            static = True
        elif arg == "--bunny":
            object_name = "bunny"
        elif arg == "--duck":
            object_name = "duck"
        elif arg == "--vader":
            object_name = "vader"
            
    run(static, object_name)
