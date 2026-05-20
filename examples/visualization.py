"""
Visualization panels, segment builders, and animation for the SOM.

Provides four-panel figures:

1. **Data panel** — input data colored by label with BMU projection lines.
2. **SOM panel** — weight grid in data space, nodes colored by U-Matrix.
3. **Activation panel** — BMU-frequency heatmap.
4. **3D panel** — for 3D datasets shows the point cloud + animated SOM mesh;
   for 2D datasets shows the U-Matrix as a surface plot.

All panels include publication-quality legends and colour bars.  The 2D
panels support matplotlib blitting for faster animation; the 3D panel
redraws fully each frame (matplotlib limitation).
"""

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from examples.datasets import PALETTE
from zsom import SOM

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.colorbar import Colorbar
    from matplotlib.figure import Figure, SubFigure
    from matplotlib.cm import ScalarMappable
    from mpl_toolkits.mplot3d import Axes3D


# ---------------------------------------------------------------------------
# Internal constants
# ---------------------------------------------------------------------------

_LABEL_NAMES: dict[str, list[str]] = {
    "clusters": ["Cluster A", "Cluster B", "Cluster C"],
    "ring": ["Inner ring", "Outer ring"],
    "swiss": ["Strip 1", "Strip 2", "Strip 3"],
    "grid": ["Group 1", "Group 2", "Group 3"],
}

_COLORMAP_UMATRIX = "YlOrRd"
_COLORMAP_HEATMAP = "YlOrBr"
_COLOR_GRID_EDGES = "#555555"
_COLOR_MESH_EDGES = "#666666"
_COLOR_BMU_LINES = "#aaaaaa"
_COLOR_SOM_NODES = "#E85D2C"


# ---------------------------------------------------------------------------
# Grid segment builders  (vectorized — no Python loops for square topology)
# ---------------------------------------------------------------------------


def _build_grid_segments_2d(som: SOM) -> np.ndarray:
    """Build all 2D grid edge segments as an ``(E, 2, 2)`` array.

    Uses array slicing to avoid Python loops for square topology.
    Falls back to explicit iteration for hex topology.

    Args:
        som: SOM instance (uses first 2 weight dimensions).

    Returns:
        Float array of shape ``(E, 2, 2)`` for
        :class:`~matplotlib.collections.LineCollection`.
    """
    if som.topology == "hex":
        segments: list[list[np.ndarray]] = []
        for i in range(som.w):
            for j in range(som.h):
                for ni, nj in som._get_hex_neighbors(i, j):
                    if (ni, nj) > (i, j):
                        segments.append(
                            [som.weights[i, j, :2], som.weights[ni, nj, :2]]
                        )
        return np.array(segments) if segments else np.empty((0, 2, 2))

    horizontal = np.stack([som.weights[:-1, :, :2], som.weights[1:, :, :2]], axis=2)
    vertical = np.stack([som.weights[:, :-1, :2], som.weights[:, 1:, :2]], axis=2)
    return np.vstack([horizontal.reshape(-1, 2, 2), vertical.reshape(-1, 2, 2)])


def _build_grid_segments_3d(som: SOM) -> np.ndarray:
    """Build all 3D grid edge segments as an ``(E, 2, 3)`` array.

    Args:
        som: SOM instance (uses all 3 weight dimensions).

    Returns:
        Float array of shape ``(E, 2, 3)`` for
        :class:`~mpl_toolkits.mplot3d.art3d.Line3DCollection`.
    """
    horizontal = np.stack([som.weights[:-1, :], som.weights[1:, :]], axis=2)
    vertical = np.stack([som.weights[:, :-1], som.weights[:, 1:]], axis=2)
    return np.vstack(
        [
            horizontal.reshape(-1, 2, som.input_dim),
            vertical.reshape(-1, 2, som.input_dim),
        ]
    )


# ---------------------------------------------------------------------------
# Colorbar helper
# ---------------------------------------------------------------------------


# Tag names stored on the axes object
_CBAR_ATTR = "_zsom_colorbar"


def _attach_or_update_colorbar(
    fig: "Figure | SubFigure",
    axes: "Axes",
    mappable: "ScalarMappable",
    label: str,
) -> None:
    """Create a colour bar on first call; update it on subsequent calls.

    Avoids removing and re-adding axes every frame, which causes
    progressive layout shrinkage during animation.

    Args:
        fig:      Parent figure.
        axes:     The data axes the colour bar describes.
        mappable: The :class:`~matplotlib.cm.ScalarMappable` to map.
        label:    Axis label for the colour bar.
    """
    existing: "Colorbar | None" = getattr(axes, _CBAR_ATTR, None)
    if existing is not None:
        existing.update_normal(mappable)
        return

    colorbar = fig.colorbar(mappable, ax=axes, fraction=0.046, pad=0.04)
    colorbar.set_label(label, fontsize=7)
    colorbar.ax.tick_params(labelsize=6)
    setattr(axes, _CBAR_ATTR, colorbar)


# ---------------------------------------------------------------------------
# Legend helpers
# ---------------------------------------------------------------------------


def _resolve_label_name(label: int, dataset_name: str | None) -> str:
    """Return a human-readable name for an integer class label.

    Args:
        label:        Integer class index.
        dataset_name: Dataset key used to look up ``_LABEL_NAMES``.

    Returns:
        Display name string.
    """
    if dataset_name == "bunny":
        return "Bunny surface"
    names = _LABEL_NAMES.get(dataset_name or "", [])
    if label < len(names):
        return names[label]
    return f"Class {label}"


def _build_scatter_legend_handle(label: int, name: str) -> Line2D:
    """Create a single circular legend proxy handle.

    Args:
        label: Class index used to pick the palette colour.
        name:  Display label string.

    Returns:
        :class:`~matplotlib.lines.Line2D` proxy.
    """
    return Line2D(
        [],
        [],
        marker="o",
        color="none",
        markerfacecolor=PALETTE[label % len(PALETTE)],
        markersize=6,
        label=name,
        linewidth=0,
    )


def _build_class_legend_handles(
    labels: np.ndarray,
    dataset_name: str | None = None,
) -> list[Line2D]:
    """Build one legend handle per unique class label.

    Args:
        labels:       Integer class labels of shape ``(n,)``.
        dataset_name: Optional dataset key for human-readable names.

    Returns:
        List of :class:`~matplotlib.lines.Line2D` proxy handles.
    """
    return [
        _build_scatter_legend_handle(
            int(lbl), _resolve_label_name(int(lbl), dataset_name)
        )
        for lbl in np.unique(labels)
    ]


# ---------------------------------------------------------------------------
# Panel renderers
# ---------------------------------------------------------------------------


def plot_data_panel(
    ax: Axes,
    pts: np.ndarray,
    labels: np.ndarray,
    bmus: np.ndarray | None,
    som: SOM | None,
    *,
    dataset_name: str | None = None,
) -> None:
    """Render the input data with optional BMU projection lines.

    Args:
        ax:           Matplotlib Axes; cleared on entry.
        pts:          Input points of shape ``(n, d)``; only first 2 dims used.
        labels:       Integer class labels of shape ``(n,)``.
        bmus:         Precomputed BMU indices ``(n, 2)``.  ``None`` → no lines.
        som:          SOM instance used to look up BMU weight positions.
        dataset_name: Optional dataset key for richer legend labels.
    """
    ax.clear()
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_aspect("equal")
    ax.set_title("Input Data + BMU Projections", fontsize=10, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])

    if bmus is not None and som is not None:
        bmu_positions = som.weights[bmus[:, 0], bmus[:, 1], :2]
        projection_segments = np.stack([pts[:, :2], bmu_positions], axis=1)
        ax.add_collection(
            LineCollection(
                projection_segments,  # type: ignore[arg-type]
                colors=_COLOR_BMU_LINES,
                lw=0.5,
                alpha=0.3,
            )
        )

    for label in np.unique(labels):
        mask = labels == label
        ax.scatter(
            pts[mask, 0],
            pts[mask, 1],
            c=PALETTE[label % len(PALETTE)],
            s=14,
            alpha=0.75,
            linewidths=0,
            zorder=2,
        )

    legend_handles = _build_class_legend_handles(labels, dataset_name)
    if bmus is not None:
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color=_COLOR_BMU_LINES,
                lw=1.0,
                alpha=0.6,
                label="BMU projection",
            )
        )
    ax.legend(
        handles=legend_handles,
        loc="upper left",
        fontsize=7,
        framealpha=0.7,
        handletextpad=0.4,
        borderpad=0.3,
    )


def plot_som_panel(ax: Axes, som: SOM) -> None:
    """Render the SOM weight grid colored by U-Matrix values.

    Args:
        ax:  Matplotlib Axes; cleared on entry.
        som: SOM instance to render.
    """
    ax.clear()
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_aspect("equal")
    ax.set_title("Weight Grid — U-Matrix", fontsize=10, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])

    umatrix = som.get_u_matrix()
    umatrix_norm = umatrix / (umatrix.max() + 1e-9)
    node_marker = "H" if som.topology == "hex" else "s"

    scatter = ax.scatter(
        som.weights[:, :, 0].ravel(),
        som.weights[:, :, 1].ravel(),
        c=umatrix_norm.ravel(),
        cmap=_COLORMAP_UMATRIX,
        marker=node_marker,
        s=20,
        vmin=0,
        vmax=1,
        zorder=3,
        linewidths=0,
    )

    grid_segments = _build_grid_segments_2d(som)
    if len(grid_segments):
        ax.add_collection(
            LineCollection(
                grid_segments,  # type: ignore[arg-type]
                colors=_COLOR_GRID_EDGES,
                lw=0.8,
                alpha=0.45,
                zorder=2,
            )
        )

    _attach_or_update_colorbar(ax.figure, ax, scatter, "U-Matrix (norm.)")


def plot_activation_panel(ax: Axes, som: SOM, bmus: np.ndarray) -> None:
    """Render the activation heatmap from precomputed BMU assignments.

    Args:
        ax:   Matplotlib Axes; cleared on entry.
        som:  SOM instance (provides grid dimensions).
        bmus: Precomputed BMU indices ``(n, 2)`` from :meth:`SOM.get_bmus`.
    """
    ax.clear()
    ax.set_title("Activation Heatmap — BMU Frequency", fontsize=10, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])

    heatmap = som.get_activation_map(bmus)
    image = ax.imshow(
        heatmap.T,
        origin="lower",
        cmap=_COLORMAP_HEATMAP,
        aspect="auto",
        interpolation="nearest",
    )

    _attach_or_update_colorbar(ax.figure, ax, image, "Hit count")


def plot_3d_panel(
    ax: Axes3D,
    pts: np.ndarray,
    labels: np.ndarray,
    som: SOM,
    bmus: np.ndarray | None,
    *,
    dataset_name: str | None = None,
    view_3d_from_top: bool = False,
) -> None:
    """Render the 3D view panel.

    For 3D input: point cloud (subsampled) + SOM mesh colored by U-Matrix.
    For 2D input: U-Matrix rendered as a surface plot.

    Args:
        ax:               Matplotlib ``Axes3D`` instance; cleared on entry.
        pts:              Input array of shape ``(n, d)``.
        labels:           Integer class labels of shape ``(n,)``.
        som:              SOM instance.
        bmus:             Precomputed BMU indices ``(n, 2)``; unused for 2D.
        dataset_name:     Optional dataset key for richer legend labels.
        view_3d_from_top: When ``True``, sets view to directly overhead.
    """
    ax.clear()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([]) # type: ignore
    if view_3d_from_top:
        ax.view_init(elev=90, azim=-90)

    if pts.shape[1] == 3:
        _render_3d_point_cloud_panel(ax, pts, labels, som, dataset_name)
    else:
        _render_umatrix_surface_panel(ax, som)


def _render_3d_point_cloud_panel(
    ax: Axes3D,
    pts: np.ndarray,
    labels: np.ndarray,
    som: SOM,
    dataset_name: str | None,
) -> None:
    """Render point cloud, SOM mesh, and legend into a 3D axes.

    Args:
        ax:           ``Axes3D`` instance.
        pts:          Input points of shape ``(n, 3)``.
        labels:       Integer class labels of shape ``(n,)``.
        som:          SOM instance.
        dataset_name: Optional dataset key for legend labels.
    """
    ax.set_title("3D Data + SOM Mesh", fontsize=10, fontweight="bold")

    # Subsample large clouds to keep rendering fast
    subsample_step = max(1, len(pts) // 2000)
    for label in np.unique(labels):
        mask = labels == label
        visible = pts[mask][::subsample_step]
        ax.scatter(
            visible[:, 0],
            visible[:, 1],
            visible[:, 2],  # type: ignore[arg-type]
            c=PALETTE[label % len(PALETTE)],
            s=12,
            alpha=0.4,
            linewidths=0,
        )

    umatrix = som.get_u_matrix()
    umatrix_norm = (umatrix / (umatrix.max() + 1e-9)).ravel()
    node_marker = "H" if som.topology == "hex" else "s"

    ax.scatter(
        som.weights[:, :, 0].ravel(),
        som.weights[:, :, 1].ravel(),
        som.weights[:, :, 2].ravel(),  # type: ignore[arg-type]
        c=umatrix_norm,
        cmap=_COLORMAP_UMATRIX,
        marker=node_marker,
        s=22,
        vmin=0,
        vmax=1,
        zorder=3,
        linewidths=0,
    )

    mesh_segments = _build_grid_segments_3d(som)
    ax.add_collection(
        Line3DCollection(
            mesh_segments,  # type: ignore[arg-type]
            colors=_COLOR_MESH_EDGES,
            lw=0.7,
            alpha=0.45,
        )
    )

    legend_handles = _build_class_legend_handles(labels, dataset_name)
    legend_handles.append(
        Line2D(
            [],
            [],
            marker="s",
            color="none",
            markerfacecolor=_COLOR_SOM_NODES,
            markersize=6,
            label="SOM nodes (U-Matrix)",
            linewidth=0,
        )
    )
    legend_handles.append(
        Line2D(
            [0],
            [0],
            color=_COLOR_MESH_EDGES,
            lw=1.2,
            alpha=0.7,
            label="SOM mesh edges",
        )
    )
    ax.legend(
        handles=legend_handles,
        loc="upper left",
        fontsize=6,
        framealpha=0.7,
        handletextpad=0.3,
        borderpad=0.3,
    )


def _render_umatrix_surface_panel(ax: Axes3D, som: SOM) -> None:
    """Render the U-Matrix as a 3D surface for 2D SOMs.

    Args:
        ax:  ``Axes3D`` instance.
        som: SOM instance.
    """
    ax.set_title("U-Matrix Surface", fontsize=10, fontweight="bold")

    umatrix = som.get_u_matrix()
    xi, yi = np.meshgrid(np.arange(som.w), np.arange(som.h), indexing="ij")
    ax.plot_surface(
        xi / som.w,
        yi / som.h,
        umatrix,
        cmap=_COLORMAP_UMATRIX,
        alpha=0.85,
        linewidth=0,
    )


# ---------------------------------------------------------------------------
# Figure factory and frame orchestrator
# ---------------------------------------------------------------------------


def _make_figure() -> tuple[Figure, Axes, Axes, Axes, Axes3D]:
    """Create a 2×2 figure with a 3D bottom-right subplot.

    Returns:
        ``(fig, ax_data, ax_som, ax_heatmap, ax_3d)`` tuple.
    """
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    fig = plt.figure(figsize=(11, 9))
    ax_data = fig.add_subplot(2, 2, 1)
    ax_som = fig.add_subplot(2, 2, 2)
    ax_heatmap = fig.add_subplot(2, 2, 3)
    ax_3d = fig.add_subplot(2, 2, 4, projection="3d")
    return fig, ax_data, ax_som, ax_heatmap, ax_3d


def _draw_frame(
    fig: Figure,
    ax_data: Axes,
    ax_som: Axes,
    ax_heatmap: Axes,
    ax_3d: Axes3D,
    som: SOM,
    pts: np.ndarray,
    labels: np.ndarray,
    epoch_label: str,
    *,
    dataset_name: str | None = None,
    view_3d_from_top: bool = False,
) -> None:
    """Redraw all four panels for one animation frame or a static snapshot.

    Computes BMU assignments once and shares the result across panels that
    need it, avoiding redundant distance broadcasts per frame.

    Args:
        fig:              Parent figure.
        ax_data:          Data panel axes.
        ax_som:           SOM weight grid axes.
        ax_heatmap:       Activation heatmap axes.
        ax_3d:            3D or U-Matrix surface axes.
        som:              Current SOM snapshot.
        pts:              Input array of shape ``(n, d)``.
        labels:           Integer class labels of shape ``(n,)``.
        epoch_label:      String shown in the figure ``suptitle``.
        dataset_name:     Optional dataset key for richer legend labels.
        view_3d_from_top: Pass through to :func:`plot_3d_panel`.
    """
    fig.suptitle(
        f"Self-Organizing Map — {epoch_label}",
        fontsize=13,
        fontweight="bold",
    )

    bmus = som.get_bmus(pts)  # computed once, shared across panels

    plot_data_panel(ax_data, pts, labels, bmus, som, dataset_name=dataset_name)
    plot_som_panel(ax_som, som)
    plot_activation_panel(ax_heatmap, som, bmus)
    plot_3d_panel(
        ax_3d,
        pts,
        labels,
        som,
        bmus,
        dataset_name=dataset_name,
        view_3d_from_top=view_3d_from_top,
    )


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def plot_static(
    som: SOM,
    pts: np.ndarray,
    labels: np.ndarray,
    *,
    dataset_name: str | None = None,
    filepath: str | None = None,
    view_3d_from_top: bool = False,
) -> None:
    """Display a static 2×2 figure of the final trained SOM state.

    Args:
        som:              Fully trained SOM instance.
        pts:              Input points of shape ``(n, d)``.
        labels:           Integer class labels of shape ``(n,)``.
        dataset_name:     Optional dataset key for richer legend labels.
        filepath:         Optional path to save the figure as an image file.
        view_3d_from_top: Render the 3D panel from directly overhead.
    """
    fig, ax_data, ax_som, ax_heatmap, ax_3d = _make_figure()
    _draw_frame(
        fig,
        ax_data,
        ax_som,
        ax_heatmap,
        ax_3d,
        som,
        pts,
        labels,
        "final",
        dataset_name=dataset_name,
        view_3d_from_top=view_3d_from_top,
    )
    plt.tight_layout()
    if filepath:
        plt.savefig(filepath, dpi=150, bbox_inches="tight", transparent=False)
        plt.close(fig)
    else:
        plt.show()


def plot_animated(
    snapshots: list[SOM],
    pts: np.ndarray,
    labels: np.ndarray,
    epochs: int,
    snapshot_every: int,
    *,
    dataset_name: str | None = None,
) -> FuncAnimation:
    """Replay the training history as a 4-panel animation.

    The 3D panel is always fully redrawn because matplotlib's ``Axes3D``
    does not support blitting natively.

    Args:
        snapshots:      Ordered SOM snapshots from :meth:`SOM.fit`.
        pts:            Input array of shape ``(n, d)``.
        labels:         Integer class labels of shape ``(n,)``.
        epochs:         Total training epochs (used only for title display).
        snapshot_every: Epoch interval between consecutive snapshots.
        dataset_name:   Optional dataset key for richer legend labels.

    Returns:
        :class:`~matplotlib.animation.FuncAnimation` instance.  Keep a
        reference to prevent premature garbage collection.
    """
    fig, ax_data, ax_som, ax_heatmap, ax_3d = _make_figure()
    plt.tight_layout()

    def _update_frame(frame_index: int) -> None:
        snapshot = snapshots[frame_index]
        current_epoch = min(frame_index * snapshot_every, epochs - 1)
        _draw_frame(
            fig,
            ax_data,
            ax_som,
            ax_heatmap,
            ax_3d,
            snapshot,
            pts,
            labels,
            f"epoch {current_epoch + 1}/{epochs}",
            dataset_name=dataset_name,
        )

    animation = FuncAnimation(
        fig,
        _update_frame, # type: ignore
        frames=len(snapshots),
        interval=40,
        repeat=False,
        blit=False,  # ty:ignore[invalid-argument-type]
    )
    plt.show()
    return animation
