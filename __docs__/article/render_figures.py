"""Generate article figures for ARTICLE.md."""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

OUTPUT_DIR = Path(__file__).parent
METRICS_DIR = OUTPUT_DIR / "metrics"

mpl.use("Agg")
plt.style.use("dark_background")

mpl.rcParams.update(
    {
        "figure.dpi": 160,
        "savefig.dpi": 160,
        "font.size": 9,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "axes.grid": True,
        "grid.linestyle": ":",
        "grid.alpha": 0.4,
        "grid.color": "0.5",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.facecolor": "#111111",
        "figure.facecolor": "#0b0b0b",
        "savefig.facecolor": "#0b0b0b",
        "text.color": "#e6e6e6",
        "axes.labelcolor": "#e6e6e6",
        "xtick.color": "#cfcfcf",
        "ytick.color": "#cfcfcf",
    }
)


def save(fig: plt.Figure, name: str) -> None:
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / name, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    # 1) SOM grid schematic
    fig, ax = plt.subplots(figsize=(3.2, 2.6))
    ax.set_title("SOM grid (w x h)")
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for i in range(3):
        for j in range(3):
            ax.plot(i, j, "ko", ms=4)
            ax.text(i + 0.06, j + 0.06, f"({i},{j})", fontsize=7)
    for i in range(3):
        ax.plot([i, i], [0, 2], color="0.6", lw=0.8)
        ax.plot([0, 2], [i, i], color="0.6", lw=0.8)
    ax.set_xlim(-0.3, 2.6)
    ax.set_ylim(-0.3, 2.6)
    save(fig, "som_grid.png")

    # 2) Neighborhood Gaussian profiles
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    ax.set_title("Neighborhood kernel")
    x = np.linspace(-3, 3, 400)
    for sigma, label in [(3.0, "early (sigma=3.0)"), (1.0, "late (sigma=1.0)")]:
        y = np.exp(-0.5 * (x / sigma) ** 2)
        ax.plot(x, y, label=label)
    ax.axvline(0, color="0.7", lw=0.8)
    ax.set_xlabel("grid distance")
    ax.set_ylabel("h(i,j)")
    ax.legend(frameon=False, fontsize=7)
    save(fig, "neighborhood_gaussian.png")

# 3) BMU update vectors
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    ax.set_title("BMU update vectors", fontsize=9, pad=8)
    ax.set_aspect("equal")

    # -- coordinates --
    w_bmu    = np.array([0.41, 0.47])
    x_sample = np.array([0.65, 0.26])
    neighbor = np.array([0.21, 0.55])

    # neighborhood radius indicator
    circle = plt.Circle(w_bmu, 0.22, color="0.75", fill=False,
                        linestyle="--", linewidth=0.8, zorder=1)
    ax.add_patch(circle)
    ax.text(w_bmu[0] + 0.16, w_bmu[1] - 0.16, "σ (neighborhood)",
            fontsize=6, color="0.6", ha="left")

    # faint grid
    for v in [0.2, 0.4, 0.6, 0.8]:
        ax.axhline(v, color="0.9", linewidth=0.4, zorder=0)
        ax.axvline(v, color="0.9", linewidth=0.4, zorder=0)

    # -- arrows --
    # strong pull: W_bmu → x  (teal, solid, thick)
    ax.annotate(
        "",
        xy=x_sample,
        xytext=w_bmu,
        arrowprops=dict(arrowstyle="->", lw=2.0, color="#1D9E75"),
        zorder=3,
    )
    # weak pull: neighbor → x  (gray, dashed, thin)
    ax.annotate(
        "",
        xy=x_sample,
        xytext=neighbor,
        arrowprops=dict(arrowstyle="->", lw=1.0, color="0.55",
                        linestyle="dashed"),
        zorder=3,
    )

    # -- nodes --
    ax.scatter([w_bmu[0]],    [w_bmu[1]],    s=60,  color="#1D9E75",
               zorder=4, edgecolors="#0F6E56", linewidths=0.8)
    ax.scatter([neighbor[0]], [neighbor[1]], s=45,  color="0.72",
               zorder=4, edgecolors="0.45", linewidths=0.8)
    ax.scatter([x_sample[0]], [x_sample[1]], s=70,  color="#D85A30",
               marker="X", zorder=4, edgecolors="#993C1D", linewidths=0.8)

    # -- labels (boxed) --
    box_style = dict(boxstyle="round,pad=0.25", fc="white", ec="0.8", lw=0.5)
    ax.text(w_bmu[0] - 0.02, w_bmu[1] + 0.05, "W_bmu",
            fontsize=7, ha="center", bbox=box_style)
    ax.text(x_sample[0] + 0.02, x_sample[1] - 0.05, "x (input)",
            fontsize=7, ha="left", bbox=box_style)
    ax.text(neighbor[0] - 0.02, neighbor[1] + 0.05, "neighbor",
            fontsize=7, ha="center", bbox=box_style)

    # -- legend --
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color="#1D9E75", lw=2.0, label="BMU pull"),
        Line2D([0], [0], color="0.55",    lw=1.0, linestyle="--",
               label="neighbor pull"),
    ]
    ax.legend(handles=legend_elements, fontsize=6.5, loc="lower right",
              framealpha=0.9, edgecolor="0.8")

    ax.set_xlim(0.05, 0.85)
    ax.set_ylim(0.10, 0.75)
    ax.set_xlabel("dim 1", fontsize=8)
    ax.set_ylabel("dim 2", fontsize=8)
    ax.tick_params(labelsize=7)
    save(fig, "bmu_update_vectors.png")

# 4) Decay schedules — linear vs exponential
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6), sharey=False)
    fig.suptitle("Decay schedules", fontsize=9, y=1.02)

    t      = np.linspace(0, 1, 200)
    epochs = np.linspace(0, 1, 200)   # ep / T  (normalised)
    eta0, sigma0 = 0.5, 6.0
    tau    = 1 / 3.0                  # τ = T/3, so ep/τ = 3·(ep/T)

    colors = {"eta": "#1D9E75", "sigma": "#378ADD"}

    # -- Left: linear --
    ax = axes[0]
    ax.set_title("Linear  $\\eta(t)=\\eta_0(1-t)$", fontsize=8)
    ax.plot(t, eta0   * (1 - t),        color=colors["eta"],   lw=1.8, label="$\\eta(t)$")
    ax.plot(t, sigma0 * (1 - t),        color=colors["sigma"],  lw=1.8, label="$\\sigma(t)$")
    ax.axvline(0.5, color="0.75", lw=0.7, linestyle=":")
    ax.text(0.52, sigma0 * 0.55, "fine-tuning\nphase", fontsize=6, color="0.5")
    ax.text(0.01, sigma0 * 0.55, "global\nphase", fontsize=6, color="0.5")
    ax.set_xlabel("$t\\ /\\ T$", fontsize=8)
    ax.set_ylabel("value", fontsize=8)
    ax.legend(frameon=False, fontsize=7)
    ax.tick_params(labelsize=7)

    # -- Right: exponential --
    ax = axes[1]
    ax.set_title("Exponential  $\\eta(t)=\\eta_0\\,e^{-t/\\tau}$", fontsize=8)
    ax.plot(t, eta0   * np.exp(-epochs / tau), color=colors["eta"],   lw=1.8, label="$\\eta(t)$")
    ax.plot(t, sigma0 * np.exp(-epochs / tau), color=colors["sigma"],  lw=1.8, label="$\\sigma(t)$")
    ax.axvline(0.5, color="0.75", lw=0.7, linestyle=":")
    ax.text(0.52, sigma0 * 0.55, "fine-tuning\nphase", fontsize=6, color="0.5")
    ax.text(0.01, sigma0 * 0.55, "global\nphase", fontsize=6, color="0.5")
    ax.set_xlabel("$t\\ /\\ T$", fontsize=8)
    ax.legend(frameon=False, fontsize=7)
    ax.tick_params(labelsize=7)

    # -- Overlay comparison on both panels (dashed ghost of the other) --
    axes[0].plot(t, eta0   * np.exp(-epochs / tau), color=colors["eta"],   lw=0.9,
                 linestyle="--", alpha=0.35, label="_exp ghost")
    axes[0].plot(t, sigma0 * np.exp(-epochs / tau), color=colors["sigma"],  lw=0.9,
                 linestyle="--", alpha=0.35)
    axes[1].plot(t, eta0   * (1 - t),               color=colors["eta"],   lw=0.9,
                 linestyle="--", alpha=0.35, label="_lin ghost")
    axes[1].plot(t, sigma0 * (1 - t),               color=colors["sigma"],  lw=0.9,
                 linestyle="--", alpha=0.35)

    fig.tight_layout()
    save(fig, "decay_schedule.png")

    # 5) Cosine vs Euclidean view
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.6), sharey=True)
    a = np.array([1.0, 0.0])
    b = np.array([0.5, 0.5])
    c = np.array([2.0, 0.0])
    for axis, title in zip(axes, ["Euclidean", "Cosine"]):
        axis.set_title(title)
        axis.set_aspect("equal")
        axis.scatter([a[0], b[0], c[0]], [a[1], b[1], c[1]], c=["k", "k", "k"], s=20)
        axis.text(a[0] + 0.05, a[1] - 0.06, "A", fontsize=7)
        axis.text(b[0] + 0.05, b[1] + 0.02, "B", fontsize=7)
        axis.text(c[0] + 0.05, c[1] - 0.06, "C", fontsize=7)
        axis.set_xlim(0, 2.2)
        axis.set_ylim(-0.2, 1.2)
        axis.set_xlabel("x")
    axes[0].set_ylabel("y")
    axes[0].plot([a[0], c[0]], [a[1], c[1]], "--", color="0.8", lw=0.8)
    axes[1].plot([0, 2.0], [0, 0.0], "--", color="0.8", lw=0.8)
    axes[1].plot([0, 0.5], [0, 0.5], "--", color="0.8", lw=0.8)
    save(fig, "cosine_vs_euclidean.png")

# 6) Lp unit balls
    fig, ax = plt.subplots(figsize=(4.2, 3.4))
    ax.set_title("$L^p$ unit balls  ($\\|\\mathbf{x}\\|_p = 1$)", fontsize=9, pad=8)
    ax.set_aspect("equal")
    ax.set_xlim(-1.35, 1.35)
    ax.set_ylim(-1.35, 1.35)
    ax.set_xlabel("$x_1$", fontsize=8)
    ax.set_ylabel("$x_2$", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.axhline(0, color="0.85", linewidth=0.6, zorder=0)
    ax.axvline(0, color="0.85", linewidth=0.6, zorder=0)
    ax.grid(color="0.93", linewidth=0.4, zorder=0)

    # -- color palette (explicit, never relies on cycle order) --
    colors = {"p1": "#D85A30", "p2": "#1D9E75", "p3": "#378ADD", "pinf": "#888780"}

    # p = 1  (diamond)
    ax.plot([0, 1, 0, -1, 0], [1, 0, -1, 0, 1],
            color=colors["p1"], lw=1.6, label="$p=1$")

    # p = 2  (circle)
    theta = np.linspace(0, 2 * np.pi, 400)
    ax.plot(np.cos(theta), np.sin(theta),
            color=colors["p2"], lw=1.6, label="$p=2$")

    # p = 3  (superellipse) — single continuous loop avoids color-cycle split
    x3 = np.linspace(-1, 1, 800)
    y3 = np.clip(1 - np.abs(x3) ** 3, 0, None) ** (1 / 3)
    ax.plot(np.concatenate([x3, x3[::-1]]),
            np.concatenate([y3, -y3[::-1]]),
            color=colors["p3"], lw=1.6, label="$p=3$")

    # p = inf  (square) — closed rectangle via array, not 5 magic points
    sq = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [-1, -1]])
    ax.plot(sq[:, 0], sq[:, 1],
            color=colors["pinf"], lw=1.6, label="$p=\\infty$")

    ax.legend(frameon=False, fontsize=7.5, loc="lower left",
              handlelength=1.4, labelspacing=0.3)
    save(fig, "lp_unit_balls.png")
    
    # 7) PCA pipeline diagram
    fig, ax = plt.subplots(figsize=(4.6, 2.6))
    ax.set_title("PCA initialization pipeline")
    ax.axis("off")
    steps = ["Center X", "SVD", "Project to PC1/PC2", "Build lattice", "Map back to R^d"]
    for i, step in enumerate(steps):
        y = 1 - i * 0.18
        rect = Rectangle((0.05, y - 0.08), 0.9, 0.12, fill=False, lw=0.8)
        ax.add_patch(rect)
        ax.text(0.07, y - 0.02, f"{i + 1}. {step}", fontsize=8)
        if i < len(steps) - 1:
            ax.annotate(
                "",
                xy=(0.5, y - 0.1),
                xytext=(0.5, y - 0.18),
                arrowprops=dict(arrowstyle="->", lw=0.8),
            )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    save(fig, "pca_pipeline.png")

    # 8) Random vs PCA init
    fig, axes = plt.subplots(1, 2, figsize=(6.0, 2.6), sharey=True)
    np.random.seed(0)
    axes[0].set_title("Random init")
    axes[0].scatter(np.random.rand(100), np.random.rand(100), s=8, c="#e6e6e6", alpha=0.6)
    axes[1].set_title("PCA init")
    cloud = np.random.multivariate_normal([0.6, 0.4], [[0.02, 0.015], [0.015, 0.02]], size=150)
    axes[1].scatter(cloud[:, 0], cloud[:, 1], s=8, c="0.7", alpha=0.7)
    line = np.linspace(0.1, 0.9, 20)
    axes[1].scatter(line, 0.3 + 0.4 * line, s=10, c="#e6e6e6")
    for axis in axes:
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
        axis.set_aspect("equal")
        axis.set_xlabel("x")
    axes[0].set_ylabel("y")
    save(fig, "pca_init_vs_random.png")

    # 9) MQE curve
    fig, ax = plt.subplots(figsize=(3.6, 2.4))
    ax.set_title("MQE convergence")
    t = np.linspace(0, 100, 200)
    curve = np.exp(-t / 18) + 0.05 + 0.02 * np.sin(t / 8)
    ax.plot(t, curve, color="#e6e6e6")
    ax.set_xlabel("epoch")
    ax.set_ylabel("MQE")
    save(fig, "mqe_curve.png")

    # 10) Topology selection table
    fig, ax = plt.subplots(figsize=(4.4, 2.4))
    ax.axis("off")
    rows = [
        ["Clusters (2D)", "OK", "Overkill"],
        ["Rings (2D)", "OK", "Better"],
        ["Swiss roll (2D)", "OK", "Marginal"],
        ["Bunny/Duck (3D)", "OK", "Recommended"],
        ["Color space", "OK", "Recommended"],
        ["Time-series", "OK", "Overkill"],
    ]
    col_labels = ["Dataset", "Square", "Hex"]
    table = ax.table(cellText=rows, colLabels=col_labels, loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1, 1.2)
    ax.set_title("Topology selection", pad=8)
    save(fig, "topology_table.png")

    # 11) U-Matrix example grid
    fig, ax = plt.subplots(figsize=(3.0, 2.6))
    ax.set_title("U-Matrix example (3x3)")
    ax.set_xticks([])
    ax.set_yticks([])
    vals = np.array([[0.2, 0.3, 0.2], [0.25, 0.32, 0.22], [0.28, 0.35, 0.27]])
    ax.imshow(vals, cmap="magma", vmin=0, vmax=0.5)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{vals[i, j]:.2f}", ha="center", va="center", fontsize=7)
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(2.5, -0.5)
    save(fig, "umatrix_example.png")

    # 12) U-Matrix heatmap ridge
    fig, ax = plt.subplots(figsize=(3.0, 2.6))
    ax.set_title("U-Matrix heatmap")
    heat = np.ones((20, 20)) * 0.2
    heat[:, 9:11] = 0.9
    ax.imshow(heat, cmap="magma")
    ax.set_xticks([])
    ax.set_yticks([])
    save(fig, "umatrix_heatmap.png")

    # 13) U-Matrix surface
    fig = plt.figure(figsize=(3.2, 2.6))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title("U-Matrix surface")
    x = np.linspace(0, 1, 40)
    y = np.linspace(0, 1, 40)
    X, Y = np.meshgrid(x, y)
    Z = 0.3 * np.exp(-((X - 0.3) ** 2 + (Y - 0.3) ** 2) / 0.01) + 0.5 * np.exp(
        -((X - 0.7) ** 2 + (Y - 0.6) ** 2) / 0.02
    )
    ax.plot_surface(X, Y, Z, cmap="magma", linewidth=0)
    ax.xaxis.pane.set_facecolor("#0b0b0b")
    ax.yaxis.pane.set_facecolor("#0b0b0b")
    ax.zaxis.pane.set_facecolor("#0b0b0b")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    save(fig, "umatrix_surface.png")

    # 14) Activation heatmap
    fig, ax = plt.subplots(figsize=(3.0, 2.6))
    ax.set_title("Activation heatmap")
    act = np.random.rand(16, 16) * 0.6
    act[10:14, 2:6] = 0.0
    ax.imshow(act, cmap="magma")
    ax.set_xticks([])
    ax.set_yticks([])
    save(fig, "activation_heatmap.png")

    # 16) Distance metric geometry (one figure per metric)
    x = np.linspace(-1.2, 1.2, 240)
    y = np.linspace(-1.2, 1.2, 240)
    X, Y = np.meshgrid(x, y)

    def save_metric(name: str, Z: np.ndarray) -> None:
        fig, ax = plt.subplots(figsize=(3.2, 2.8))
        ax.set_title(name)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        im = ax.imshow(
            Z,
            extent=[x.min(), x.max(), y.min(), y.max()],
            origin="lower",
            cmap="magma",
        )
        ax.contour(X, Y, Z, levels=8, colors="#e6e6e6", linewidths=0.6)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(METRICS_DIR / f"{name.lower().replace(' ', '_')}.png", bbox_inches="tight")
        plt.close(fig)

    # Euclidean (L2)
    save_metric("euclidean_metric", np.sqrt(X**2 + Y**2))
    # Manhattan (L1)
    save_metric("manhattan_metric", np.abs(X) + np.abs(Y))
    # Chebyshev (L∞)
    save_metric("chebyshev_metric", np.maximum(np.abs(X), np.abs(Y)))
    # Cosine distance (1 - cos)
    denom = np.sqrt(X**2 + Y**2) + 1e-12
    cos = X / denom  # angle to x-axis unit vector
    save_metric("cosine_metric", 1.0 - cos)
    # Minkowski p=3
    p = 3.0
    save_metric("minkowski_metric", (np.abs(X) ** p + np.abs(Y) ** p) ** (1.0 / p))

    # 15) Square and hex topology images
    mpl.rcParams["axes.grid"] = False
    fig, ax = plt.subplots(figsize=(3.0, 2.6))
    ax.set_title("Square topology")
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for i in range(3):
        for j in range(3):
            ax.plot(i, j, "ko", ms=4)
    for i in range(3):
        ax.plot([0, 2], [i, i], color="0.6", lw=0.8)
        ax.plot([i, i], [0, 2], color="0.6", lw=0.8)
    ax.set_xlim(-0.3, 2.3)
    ax.set_ylim(-0.3, 2.3)
    save(fig, "square_topology.png")

    fig, ax = plt.subplots(figsize=(3.0, 2.6))
    ax.set_title("Hex topology")
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    coords = []
    for i in range(3):
        for j in range(3):
            x = i + 0.5 * (j % 2)
            y = j * np.sqrt(3) / 2
            coords.append((x, y))
            ax.plot(x, y, "ko", ms=4)
    for (x1, y1) in coords:
        for (x2, y2) in coords:
            d = np.hypot(x1 - x2, y1 - y2)
            if 0.7 < d < 1.1:
                ax.plot([x1, x2], [y1, y2], color="0.6", lw=0.6)
    ax.set_xlim(-0.3, 3.0)
    ax.set_ylim(-0.3, 2.8)
    save(fig, "hex_topology.png")


if __name__ == "__main__":
    main()