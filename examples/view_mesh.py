#!/usr/bin/env python3
"""
Quick viewer for 3-D mesh datasets.

Usage::

    python view_mesh.py                        # bunny + duck (defaults)
    python view_mesh.py bunny.obj duck.obj     # explicit paths
    python view_mesh.py --n 2000 bunny.obj     # custom point count
"""
import os
import sys
import argparse
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib.pyplot as plt
import numpy as np

from examples.datasets import load_obj_mesh, load_stl_vertices, _sample_surface

plt.style.use({
    # Base background configuration
    'figure.facecolor': '#000000',
    'figure.edgecolor': '#000000',
    'axes.facecolor': '#000000',
    'savefig.facecolor': '#000000',
    'savefig.edgecolor': '#000000',
    
    # High contrast foreground
    'axes.edgecolor': '#ffffff',
    'axes.labelcolor': '#ffffff',
    'text.color': '#ffffff',
    'xtick.color': '#ffffff',
    'ytick.color': '#ffffff',
    
    # Grid configuration for 3D depth perception
    'grid.color': '#555555',
    'grid.linestyle': '-',
    'grid.alpha': 0.8,
    
    # 3D specific settings (requires Matplotlib 3.3+)
    'axes3d.grid': True,
    'axes3d.xaxis.panecolor': '#080808',
    'axes3d.yaxis.panecolor': '#080808',
    'axes3d.zaxis.panecolor': '#080808',
})

# ---------------------------------------------------------------------------
# Mesh loading
# ---------------------------------------------------------------------------

def load_mesh(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load vertices and faces from a supported mesh file.

    Args:
        path: Path to a ``.obj`` or ``.stl`` file.

    Returns:
        ``(vertices, faces)`` with 0-based face indices.

    Raises:
        ValueError: If the extension is not supported.
    """
    suffix = path.suffix.lower()
    if suffix == ".obj":
        return load_obj_mesh(path)
    if suffix == ".stl":
        vertices = load_stl_vertices(path)
        n = (len(vertices) // 3) * 3
        faces = np.arange(n, dtype=np.int32).reshape(-1, 3)
        return vertices, faces
    raise ValueError(f"Unsupported format: '{suffix}'. Expected .obj or .stl.")


# ---------------------------------------------------------------------------
# Geometry builders
# ---------------------------------------------------------------------------

def _unique_edges(faces: np.ndarray) -> np.ndarray:
    """Extract deduplicated edge pairs from a triangular face array.

    Args:
        faces: Int array of shape ``(F, 3)`` with 0-based indices.

    Returns:
        Int array of shape ``(E, 2)`` with sorted, unique vertex pairs.
    """
    edges = np.concatenate(
        [faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [0, 2]]], axis=0
    )
    return np.unique(np.sort(edges, axis=1), axis=0)


def _edges_to_nan_separated(
    vertices: np.ndarray,
    edges: np.ndarray,
) -> np.ndarray:
    """Pack edge endpoints into a NaN-separated array for a single plot call.

    Matplotlib draws a polyline as one draw call.  Inserting NaN rows breaks
    the line between edges without requiring separate ``plot()`` calls, which
    makes rendering O(1) GPU/CPU draw calls instead of O(E).

    Layout per edge: ``[v0, v1, NaN_row]`` → shape ``(E*3, 3)``.

    Args:
        vertices: Float array of shape ``(V, 3)``.
        edges:    Int array of shape ``(E, 2)``.

    Returns:
        Float array of shape ``(E*3, 3)``.
    """
    v0  = vertices[edges[:, 0]]           # (E, 3)
    v1  = vertices[edges[:, 1]]           # (E, 3)
    nan = np.full_like(v0, np.nan)        # (E, 3)
    # Interleave: v0, v1, nan  →  (E, 3, 3)  →  (E*3, 3)
    return np.stack([v0, v1, nan], axis=1).reshape(-1, 3)


# ---------------------------------------------------------------------------
# Per-model renderer
# ---------------------------------------------------------------------------

def _render_model(
    ax,
    vertices: np.ndarray,
    faces: np.ndarray,
    pts: np.ndarray,
    title: str,
    point_color: str,
    edge_color: str,
) -> None:
    """Render a single mesh model into an ``Axes3D``.

    Args:
        ax:          ``Axes3D`` instance to draw into.
        vertices:    Normalized vertex array of shape ``(V, 3)``.
        faces:       Face index array of shape ``(F, 3)``.
        pts:         Sampled surface points of shape ``(n, 3)``.
        title:       Panel title string.
        point_color: Matplotlib colour for the scatter cloud.
        edge_color:  Matplotlib colour for the wireframe edges.
    """
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    # Sampled surface points — single scatter call
    ax.scatter(
        pts[:, 0], pts[:, 1], pts[:, 2],
        c=point_color,
        s=4,
        alpha=0.35,
        linewidths=0,
        depthshade=False,  # avoids per-point alpha recomputation
    )

    # Wireframe — single plot call via NaN-separated segments
    edges   = _unique_edges(faces)
    wire    = _edges_to_nan_separated(vertices, edges)
    ax.plot(
        wire[:, 0], wire[:, 1], wire[:, 2],
        color=edge_color,
        lw=0.4,
        alpha=0.5,
    )

    ax.set_xlabel("x", fontsize=7)
    ax.set_ylabel("y", fontsize=7)
    ax.set_zlabel("z", fontsize=7)


# ---------------------------------------------------------------------------
# Figure layout
# ---------------------------------------------------------------------------

def plot_meshes(
    paths: list[Path],
    n_points: int = 1500,
    seed: int = 42,
) -> None:
    """Plot all mesh models side-by-side in a single figure.

    Args:
        paths:    List of mesh file paths to visualize.
        n_points: Number of surface points to sample per model.
        seed:     Random seed for reproducibility.
    """
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    rng    = np.random.default_rng(seed)
    n_cols = min(len(paths), 3)
    n_rows = (len(paths) + n_cols - 1) // n_cols
    fig    = plt.figure(figsize=(5.5 * n_cols, 5 * n_rows))

    colors = ["#378ADD", "#D85A30", "#1D9E75", "#D4537E", "#BA7517", "#7F77DD"]

    for idx, path in enumerate(paths, start=1):
        vertices, faces = load_mesh(path)
        pts             = _sample_surface(vertices, faces, n_points, rng)
        color           = colors[(idx - 1) % len(colors)]

        ax = fig.add_subplot(n_rows, n_cols, idx, projection="3d")
        _render_model(
            ax,
            vertices=vertices,
            faces=faces,
            pts=pts,
            title=path.stem.capitalize(),
            point_color=color,
            edge_color="#888888",
        )

    fig.suptitle("3-D Mesh Viewer", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _default_paths() -> list[Path]:
    """Return default mesh paths relative to this script's ``data/`` sibling."""
    data_dir = Path(__file__).parent / "data"
    return [p for p in [data_dir / "bunny.obj", data_dir / "duck.obj", data_dir / "vader.obj"] if p.exists()]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "meshes",
        nargs="*",
        type=Path,
        help="Mesh files to visualize (.obj or .stl). Defaults to data/bunny.obj and data/duck.obj.",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=1500,
        dest="n_points",
        help="Surface points to sample per model (default: 1500).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42).",
    )
    args = parser.parse_args(argv)

    paths = args.meshes or _default_paths()
    if not paths:
        print("No mesh files found. Pass paths explicitly or add files to data/.")
        sys.exit(1)

    missing = [p for p in paths if not p.exists()]
    if missing:
        for p in missing:
            print(f"File not found: {p}")
        sys.exit(1)

    plot_meshes(paths, n_points=args.n_points, seed=args.seed)


if __name__ == "__main__":
    main()