"""
Built-in dataset generators for the Self-Organizing Map.

Each generator has the signature::

    (n: int, rng: np.random.Generator, *args, **kwargs)
        -> tuple[np.ndarray, np.ndarray]

The returned tuple is ``(pts, labels)`` where ``pts`` has shape ``(n, d)``
and ``labels`` has shape ``(n,)`` with integer class indices.

Available datasets
------------------
clusters
    Three isotropic Gaussian blobs (2D).  The simplest benchmark: a
    well-trained SOM should tile into three clearly separated regions.
ring
    Two concentric rings (2D).  Non-convex structure that tests the SOM's
    ability to wrap its grid around topologically non-trivial manifolds.
swiss
    2D Swiss roll projection.  A curvilinear manifold that the SOM must
    learn to *unroll* rather than just partition.
grid
    Jittered 5×5 uniform lattice (2D).  A nearly regular distribution
    that a well-trained SOM should tile evenly.
obj
    3D mesh point cloud (bunny/duck/vader). Loaded from a ``.obj`` or
    ``.stl`` mesh file via area-weighted surface sampling.
"""

import struct
from pathlib import Path
from typing import Callable, Literal

import numpy as np
from numpy.random import Generator

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

ObjectName = Literal["bunny", "duck", "vader"]
"""Supported built-in mesh names for the obj dataset."""

DatasetName = Literal["clusters", "ring", "swiss", "grid", "obj"]
"""Valid dataset names for the built-in generators."""

DatasetFunc = Callable[..., tuple[np.ndarray, np.ndarray]]
"""Signature of a dataset generator function."""

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Six-color palette shared by visualizations and examples.
PALETTE: list[str] = [
    "#378ADD",
    "#D85A30",
    "#1D9E75",
    "#D4537E",
    "#BA7517",
    "#7F77DD",
]

# ---------------------------------------------------------------------------
# Dataset generators
# ---------------------------------------------------------------------------

def make_clusters(
    n: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate three isotropic Gaussian clusters in the unit square.

    Cluster centers are fixed at ``(0.25, 0.25)``, ``(0.75, 0.25)``, and
    ``(0.50, 0.78)``.  Each point is assigned to a random center and
    perturbed with Gaussian noise (σ = 0.09).

    Args:
        n:   Number of points to generate.
        rng: NumPy Generator instance for reproducibility.

    Returns:
        ``(pts, labels)`` — points of shape ``(n, 2)`` and cluster indices
        ``{0, 1, 2}``.
    """
    centers = np.array([[0.25, 0.25], [0.75, 0.25], [0.50, 0.78]])
    labels = rng.integers(0, 3, n)
    return centers[labels] + rng.normal(0, 0.09, (n, 2)), labels


def make_rings(
    n: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate two concentric rings centered at ``(0.5, 0.5)``.

    Inner ring radius ≈ 0.18, outer ≈ 0.38, both with slight radial
    jitter (σ = 0.02).  Non-convex structure tests the SOM's ability to
    wrap its grid around a topologically non-trivial manifold.

    Args:
        n:   Number of points to generate.
        rng: NumPy Generator instance for reproducibility.

    Returns:
        ``(pts, labels)`` — points of shape ``(n, 2)`` and ring indices
        ``{0, 1}``.
    """
    labels = rng.integers(0, 2, n)
    radii = np.where(labels == 0, 0.18, 0.38) + rng.normal(0, 0.02, n)
    angles = rng.uniform(0, 2 * np.pi, n)
    pts = np.stack(
        [0.5 + np.cos(angles) * radii, 0.5 + np.sin(angles) * radii],
        axis=1,
    )
    return pts, labels


def make_swiss(
    n: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a 2D projection of the Swiss roll manifold.

    The angular parameter *t* is drawn from ``[1.5π, 4.5π]``, covering one
    and a half spiral turns.  A SOM must learn to unroll the curvilinear
    manifold.

    Args:
        n:   Number of points to generate.
        rng: NumPy Generator instance for reproducibility.

    Returns:
        ``(pts, labels)`` — points of shape ``(n, 2)`` and strip indices
        ``{0, 1, 2}``.
    """
    t = 1.5 * np.pi * (1 + 2 * rng.uniform(0, 1, n))
    strip = rng.integers(0, 3, n)
    x = (t / 5) * np.cos(t) * 0.18 + 0.5
    y = 0.15 + strip * 0.35 + rng.normal(0, 0.03, n)
    return np.stack([x, y], axis=1), strip


def make_grid(
    n: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a jittered uniform grid from a 5×5 lattice.

    Labels cycle through three groups ``(gx % 3)`` so adjacent columns
    share a color.  A well-trained SOM should tile it evenly.

    Args:
        n:   Number of points to generate.
        rng: NumPy Generator instance for reproducibility.

    Returns:
        ``(pts, labels)`` — points of shape ``(n, 2)`` and group indices
        ``{0, 1, 2}``.
    """
    gsize = 5
    gx = rng.integers(0, gsize, n)
    gy = rng.integers(0, gsize, n)
    pts = np.stack(
        [
            (gx + 0.5) / gsize + rng.normal(0, 0.03, n),
            (gy + 0.5) / gsize + rng.normal(0, 0.03, n),
        ],
        axis=1,
    )
    return pts, gx % 3


# ---------------------------------------------------------------------------
# Mesh I/O
# ---------------------------------------------------------------------------

def load_stl_vertices(path: str | Path) -> np.ndarray:
    """Load and deduplicate vertices from a binary STL file.

    Parses the binary STL format (80-byte header + triangle records),
    extracts all vertex coordinates, deduplicates them, and normalizes
    the result to ``[0, 1]^3``.

    Args:
        path: Filesystem path to the ``.stl`` file.

    Returns:
        Float array of shape ``(V, 3)`` with unique normalized vertices.

    Raises:
        ValueError: If the file is truncated or otherwise invalid.
    """
    with open(path, "rb") as f:
        f.read(80)
        n_triangles = struct.unpack("<I", f.read(4))[0]
        verts: list[list[float]] = []
        for _ in range(n_triangles):
            data = f.read(50)
            if len(data) < 50:
                raise ValueError(
                    f"Truncated STL: expected {n_triangles} triangles."
                )
            for i in range(3):
                x, y, z = struct.unpack_from("<fff", data, 12 + i * 12)
                verts.append([x, y, z])

    pts = np.unique(np.array(verts, dtype=np.float64), axis=0)
    pts -= pts.min(axis=0)
    pts /= pts.max(axis=0)
    return pts


def load_obj_mesh(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load vertices and triangular faces from a Wavefront OBJ file.

    Parses ``v `` lines for vertex coordinates and ``f `` lines for face
    indices.  Face entries may use the ``v``, ``v/vt``, ``v/vt/vn``, or
    ``v//vn`` formats — only the vertex index is used.  OBJ indices are
    1-based and are converted to 0-based here.  Non-triangular faces
    (quads, n-gons) are fan-triangulated from the first vertex.

    Vertex normal lines (``vn``) and texture coordinate lines (``vt``) are
    silently ignored — only geometry is extracted.

    Args:
        path: Filesystem path to the ``.obj`` file.

    Returns:
        ``(vertices, faces)`` where ``vertices`` has shape ``(V, 3)``
        normalized to ``[0, 1]^3``, and ``faces`` has shape ``(F, 3)``
        with 0-based vertex indices.

    Raises:
        ValueError: If no vertex or face lines are found in the file.
    """
    verts: list[list[float]] = []
    faces: list[list[int]] = []

    with open(path, "r") as f:
        for line in f:
            if line.startswith("v "):
                _, x, y, z = line.split()
                verts.append([float(x), float(y), float(z)])
            elif line.startswith("f "):
                # Strip texture/normal components: "v/vt/vn" or "v//vn" → v
                tokens = line.split()[1:]
                indices = [int(t.split("/")[0]) - 1 for t in tokens]
                # Fan-triangulate n-gons: (0,1,2), (0,2,3), ...
                for i in range(1, len(indices) - 1):
                    faces.append([indices[0], indices[i], indices[i + 1]])

    if not verts:
        raise ValueError(f"No vertices found in OBJ file: {path}")
    if not faces:
        raise ValueError(f"No faces found in OBJ file: {path}")

    pts = np.array(verts, dtype=np.float64)
    pts -= pts.min(axis=0)
    pts /= pts.max(axis=0)
    return pts, np.array(faces, dtype=np.int32)


def _sample_surface(
    vertices: np.ndarray,
    faces: np.ndarray,
    n: int,
    rng: Generator,
) -> np.ndarray:
    """Sample points uniformly on a triangular mesh surface.

    Weights each triangle by its area so denser regions of the mesh do not
    produce more points than sparse ones.  Uses barycentric interpolation
    to place each point inside its chosen triangle.

    Args:
        vertices: Float array of shape ``(V, 3)``.
        faces:    Int array of shape ``(F, 3)`` with 0-based indices.
        n:        Number of points to sample.
        rng:      NumPy ``Generator`` instance.

    Returns:
        Float array of shape ``(n, 3)``.
    """
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]

    cross = np.cross(v1 - v0, v2 - v0)
    areas = 0.5 * np.linalg.norm(cross, axis=1)
    probs = areas / areas.sum()

    chosen = rng.choice(len(faces), size=n, p=probs)
    r1 = rng.random(n)
    r2 = rng.random(n)

    # Fold points outside the triangle back in via barycentric reflection
    mask = (r1 + r2) > 1.0
    r1[mask] = 1.0 - r1[mask]
    r2[mask] = 1.0 - r2[mask]

    a, b, c = v0[chosen], v1[chosen], v2[chosen]
    return a + r1[:, np.newaxis] * (b - a) + r2[:, np.newaxis] * (c - a)


def _load_mesh_from_file(path: Path) -> tuple[np.ndarray, np.ndarray] | np.ndarray:
    """Dispatch mesh loading by file extension.

    Returns ``(vertices, faces)`` for ``.obj`` and a vertex-only array for
    ``.stl``.  The caller is responsible for routing to the appropriate
    sampling strategy.

    Args:
        path: Resolved filesystem path to a ``.obj`` or ``.stl`` file.

    Returns:
        ``(vertices, faces)`` tuple for OBJ, or ``vertices`` array for STL.

    Raises:
        ValueError: If the extension is not ``.obj`` or ``.stl``.
    """
    suffix = path.suffix.lower()
    if suffix == ".obj":
        return load_obj_mesh(path)
    if suffix == ".stl":
        return load_stl_vertices(path)
    raise ValueError(
        f"Unsupported mesh format: '{suffix}'. Expected .obj or .stl."
    )


def _sample_mesh(
    path: Path,
    n: int,
    rng: Generator,
) -> np.ndarray:
    """Load a mesh file and sample ``n`` points from its surface.

    Uses area-weighted triangle sampling for ``.obj`` files and random
    vertex sub-sampling for ``.stl`` files.

    Args:
        path: Resolved filesystem path to the mesh.
        n:    Number of points to sample.
        rng:  NumPy ``Generator`` instance.

    Returns:
        Float array of shape ``(n, 3)`` normalized to ``[0, 1]^3``.
    """
    result = _load_mesh_from_file(path)
    if isinstance(result, tuple):
        vertices, faces = result
        return _sample_surface(vertices, faces, n, rng)

    # STL: vertex-only sub-sampling
    vertices = result
    idx = rng.choice(len(vertices), size=min(n, len(vertices)), replace=False)
    return vertices[idx]


def _resolve_mesh_path(
    mesh_path: str | Path | None,
    obj: ObjectName,
) -> Path:
    """Resolve a mesh path, defaulting to ``data/{obj}.obj`` if needed.

    Args:
        mesh_path: Explicit path provided by the caller, or ``None``.
        obj:       Default mesh name when ``mesh_path`` is not provided.

    Returns:
        Resolved :class:`~pathlib.Path`.

    Raises:
        FileNotFoundError: If the resolved path does not exist.
    """
    resolved = (
        Path(mesh_path)
        if mesh_path is not None
        else Path(__file__).parent / "data" / f"{obj}.obj"
    )
    if not resolved.exists():
        raise FileNotFoundError(
            f"Mesh not found: {resolved}. "
            "Pass mesh_path= explicitly or place the file at that location."
        )
    return resolved


# ---------------------------------------------------------------------------
# Mesh-based dataset generators
# ---------------------------------------------------------------------------

def make_obj(
    n: int,
    rng: Generator,
    obj: ObjectName = "bunny",
    mesh_path: str | Path | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a point cloud from a mesh file.

    Loads the mesh from ``mesh_path`` and samples ``n`` points uniformly on
    its surface.  For ``.obj`` files, area-weighted triangle sampling with
    barycentric interpolation is used.  For ``.stl`` files, unique vertices
    are sub-sampled directly.

    ``mesh_path`` defaults to ``data/bunny.obj`` relative to this file when
    not provided, so the common call ``make_obj(n, rng, obj="bunny")`` works out of the
    box as long as the asset is present.

    Args:
        n:         Number of points to sample.
        rng:       NumPy ``Generator`` instance for reproducibility.
        obj:       Name of the default mesh to load.
        mesh_path: Path to a ``.obj`` or ``.stl`` mesh file.  Defaults to
                   ``<package_root>/data/{obj}.obj``.

    Returns:
        ``(pts, labels)`` where ``pts`` has shape ``(n, 3)`` normalized to
        ``[0, 1]^3`` and ``labels`` is an all-zero array of shape ``(n,)``.

    Raises:
        FileNotFoundError: If the resolved mesh path does not exist.
        ValueError: If the file extension is not ``.obj`` or ``.stl``.
    """
    path = _resolve_mesh_path(mesh_path, obj)
    pts  = _sample_mesh(path, n, rng)
    return pts, np.zeros(n, dtype=np.int64)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

DATASETS: dict[DatasetName, DatasetFunc] = {
    "clusters": make_clusters,
    "ring":     make_rings,
    "swiss":    make_swiss,
    "grid":     make_grid,
    "obj":      make_obj,
}
"""Registry of built-in dataset generators, keyed by canonical name."""