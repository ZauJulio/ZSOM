#!/usr/bin/env python3
"""Full CLI demo — drop-in replacement for the original ``python som.py``.

Usage::

    python examples/demo_all.py
    python examples/demo_all.py --dataset obj --obj vader --grid 16
    python examples/demo_all.py --dataset ring --topology hex
    python examples/demo_all.py --metric manhattan
    python examples/demo_all.py --static
    python examples/demo_all.py --no-numba
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse

from zsom import SOM, METRICS, dist_minkowski, DistanceFunc
from examples.datasets import DATASETS
from examples.visualization import plot_static, plot_animated

def main() -> None:
    """CLI entry point for the SOM demo.

    Generates the requested dataset, initialise a SOM with the chosen
    topology and distance metric, runs vectorized batch training, then
    displays either a static 2×2 plot or an animated replay.
    """
    parser = argparse.ArgumentParser(
        description="ZSOM — Self-Organizing Map demo"
    )
    parser.add_argument(
        "--dataset", choices=list(DATASETS), default="clusters"
    )
    # 3d object
    parser.add_argument(
        "--obj", choices=["bunny", "duck", "vader"], default="bunny"
    )
    parser.add_argument(
        "--topology", choices=["square", "hex"], default="square"
    )
    parser.add_argument(
        "--metric",
        choices=["euclidean", "manhattan", "chebyshev", "cosine", "minkowski"],
        default="euclidean",
    )
    parser.add_argument(
        "--minkowski-p",
        type=float,
        default=3.0,
        help="Minkowski exponent (used when --metric minkowski)",
    )
    parser.add_argument("--grid", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=0.5)
    parser.add_argument("--n", type=int, default=300)
    parser.add_argument("--static", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--no-numba",
        action="store_true",
        help="Disable Numba acceleration even if available",
    )
    parser.add_argument(
        "--init",
        choices=["random", "pca"],
        default="random",
        help="Weight initialization method",
    )
    parser.add_argument(
        "--adaptive-lr",
        action="store_true",
        help="Enable derivative-based adaptive learning rate",
    )
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    pts, labels = DATASETS[args.dataset](args.n, rng, args.obj)
    input_dim = pts.shape[1]

    metric_fn: DistanceFunc = (
        dist_minkowski(args.minkowski_p) # type: ignore
        if args.metric == "minkowski"
        else METRICS[args.metric]
    )

    use_numba = None if not args.no_numba else False

    som = SOM(
        args.grid,
        args.grid,
        input_dim,
        rng,
        topology=args.topology,
        metric=metric_fn,
        use_numba=use_numba,
        init=args.init,
        data=pts if args.init == "pca" else None,
    )

    snapshot_every = 4
    accel = "numba" if som.use_numba else "numpy"
    print(
        f"Training {args.grid}×{args.grid} {args.topology} SOM | "
        f"dataset={args.dataset} | metric={args.metric} | "
        f"init={args.init} | "
        f"{len(pts)} pts {input_dim}D | {args.epochs} epochs | "
        f"backend={accel}"
        + (" | adaptive_lr" if args.adaptive_lr else "")
    )
    snapshots = som.fit(
        pts, args.epochs, args.lr,
        snapshot_every=snapshot_every,
        adaptive_lr=args.adaptive_lr,
    )
    print(f"Done. Final qE={som.error_history_[-1]:.4f}")

    # Set view_from_top for 2D datasets if viewing statically
    view_3d_from_top = args.dataset in ("swiss", "grid", "clusters")

    if args.static:
        plot_static(som, pts, labels, dataset_name=args.dataset, view_3d_from_top=view_3d_from_top)
    else:
        plot_animated(
            snapshots,
            pts,
            labels,
            args.epochs,
            snapshot_every,
            dataset_name=args.dataset,
        )

if __name__ == "__main__":
    main()
