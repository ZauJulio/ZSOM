import os
import sys

import numpy as np
import matplotlib.pyplot as plt

from zsom import SOM
from zsom.metrics import dist_manhattan, dist_chebyshev

# Add project root to sys.path so 'examples' package is resolvable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from examples.datasets import make_clusters, make_rings, make_swiss, make_obj
from examples.visualization import plot_static

plt.style.use("dark_background")


def main():
    out_dir = "__docs__"
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(42)

    print("Generating clusters.png...")
    pts_c, labels_c = make_clusters(300, rng)
    som_c = SOM(16, 16, 2, rng, init="random", data=pts_c, metric=dist_manhattan, decay_schedule="exponential")
    som_c.fit(pts_c, epochs=128, learning_rate=0.6, adaptive_lr=True)
    plot_static(
        som_c,
        pts_c,
        labels_c,
        view_3d_from_top=True,
        dataset_name="clusters",
        filepath=os.path.join(out_dir, "clusters.png"),
    )

    print("Generating rings.png...")
    pts_r, labels_r = make_rings(300, rng)
    som_r = SOM(12, 12, 2, rng, metric=dist_chebyshev)
    som_r.fit(pts_r, epochs=128, learning_rate=0.6)
    plot_static(
        som_r,
        pts_r,
        labels_r,
        view_3d_from_top=True,
        dataset_name="ring",
        filepath=os.path.join(out_dir, "rings.png"),
    )

    print("Generating swiss.png...")
    pts_s, labels_s = make_swiss(300, rng)
    som_s = SOM(12, 12, 2, rng, metric=dist_manhattan)
    som_s.fit(pts_s, epochs=128, learning_rate=0.6, adaptive_lr=True)
    plot_static(
        som_s,
        pts_s,
        labels_s,
        view_3d_from_top=True,
        dataset_name="swiss",
        filepath=os.path.join(out_dir, "swiss.png"),
    )

    print("Generating bunny.png...")
    pts_b, labels_b = make_obj(600, rng, obj="bunny")
    som_b = SOM(32, 32, 3, rng, init="pca", data=pts_b, topology="hex", decay_schedule="exponential")
    som_b.fit(pts_b, epochs=1000, learning_rate=0.6)
    plot_static(
        som_b,
        pts_b,
        labels_b,
        dataset_name="bunny",
        view_3d_from_top=True,
        filepath=os.path.join(out_dir, "bunny.png"),
    )
    

    print("Generating duck.png...")
    pts_d, labels_d = make_obj(600, rng, obj="duck")
    som_d = SOM(32, 32, 3, rng, init="pca", data=pts_d, topology="hex", decay_schedule="exponential")
    som_d.fit(pts_d, epochs=1000, learning_rate=0.6)
    plot_static(
        som_d,
        pts_d,
        labels_b,
        dataset_name="duck",
        view_3d_from_top=True,
        filepath=os.path.join(out_dir, "duck.png"),
    )

    print("Generating vader.png...")
    pts_d, labels_d = make_obj(600, rng, obj="vader")
    som_d = SOM(32, 32, 3, rng, topology="hex", init="pca", data=pts_d, decay_schedule="exponential")
    som_d.fit(pts_d, epochs=1000, learning_rate=0.6)
    plot_static(
        som_d,
        pts_d,
        labels_b,
        dataset_name="vader",
        view_3d_from_top=True,
        filepath=os.path.join(out_dir, "vader.png"),
    )

    print("Done!")


if __name__ == "__main__":
    main()
