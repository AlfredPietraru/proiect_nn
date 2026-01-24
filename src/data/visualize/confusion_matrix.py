from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import seaborn as sns

from data.visualize.visualize_common import normalize_rows, wrap_text, format_series, save_figure

sns.set_theme(style="whitegrid")


def plot_heatmap(
    cm: np.ndarray, class_names: list[str], ax: Axes, 
    title: str, fmt: str, cmap: str, rotation_x: int,
    vmin: float | None = None, vmax: float | None = None
) -> None:
    """
    Plot a confusion-matrix heatmap on a given axis (rows=true, cols=pred).
    Heatmap for confusion matrix visualization.
    """
    sns.heatmap(
        cm, annot=True, fmt=fmt, cmap=cmap, vmin=vmin, vmax=vmax,
        cbar=True, square=True, linewidths=0.5, linecolor="white",
        xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=rotation_x, ha="right")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)


def plot_confusion_matrix(
    cm: np.ndarray, class_names: list[str],
    title: str = "Confusion Matrix",
    figsize: tuple[int, int] = (8, 6), show: bool = True,
    save_path: str | None = None, normalize: bool = False,
    fmt: str = "d", cmap: str = "Greens"
) -> tuple[Figure, Axes]:
    """Plot a single confusion matrix as a heatmap (rows=true, cols=pred)."""
    cm = np.asarray(cm)

    if normalize:
        cm = normalize_rows(cm)
        fmt = ".2f"

    n = len(class_names)
    if n >= 12:
        figsize = (max(figsize[0], 10), max(figsize[1], 8))
    if n >= 18:
        figsize = (max(figsize[0], 12), max(figsize[1], 10))

    rotation_x: int = 45

    fig, ax = plt.subplots(figsize=figsize)
    plot_heatmap(cm, class_names, ax, title, fmt, cmap, rotation_x)
    ax.set_aspect("equal")

    ax.tick_params(axis="x", labelsize=8)
    ax.tick_params(axis="y", labelsize=8)

    ax.set_xticks(np.arange(-0.5, cm.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, cm.shape[0], 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.8)
    ax.tick_params(which="minor", bottom=False, left=False)

    fig.tight_layout()
    if save_path is not None:
        save_figure(fig, save_path, dpi=300)
    if show:
        plt.show()

    return fig, ax


def plot_confusion_matrices_side_by_side(
    cm_left: np.ndarray, cm_right: np.ndarray, class_names: list[str],
    left_acc: float | list[float] | None, 
    right_acc: float | list[float] | None,
    left_title: str = "Model A", right_title: str = "Model B",
    left_cmap: str = "Blues", right_cmap: str = "Greens",
    figsize: tuple[int, int] = (16, 7), fmt: str = "d",
    shared_scale: bool = True, acc_max_items: int | None = None,
    show: bool = True, save_path: str | None = None
) -> tuple[Figure, list[Axes]]:
    """
    Plot two confusion matrices side by side for comparison.
    The comparasions are meant to be between two models (e.g., Model A vs. Model B)
    or two conditions (e.g., before vs. after), etc.
    """
    cm_left = normalize_rows(np.asarray(cm_left))
    cm_right = normalize_rows(np.asarray(cm_right))
    fmt = ".2f"

    left_acc_str = format_series(left_acc, decimals=4, max_items=acc_max_items)
    right_acc_str = format_series(right_acc, decimals=4, max_items=acc_max_items)

    if left_acc_str is not None:
        left_title += f"\n(Acc: [{left_acc_str}])"
    if right_acc_str is not None:
        right_title += f"\n(Acc: [{right_acc_str}])"

    left_title = wrap_text(left_title, 80)
    right_title = wrap_text(right_title, 80)

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    vmin = vmax = None
    if shared_scale:
        vmin = float(min(np.nanmin(cm_left), np.nanmin(cm_right)))
        vmax = float(max(np.nanmax(cm_left), np.nanmax(cm_right)))

    plot_heatmap(cm_left, class_names, axes[0], left_title, fmt, left_cmap, 45, vmin, vmax)
    plot_heatmap(cm_right, class_names, axes[1], right_title, fmt, right_cmap, 45, vmin, vmax)

    for ax in axes:
        ax.set_aspect("equal")
        ax.tick_params(axis="x", labelsize=8)
        ax.tick_params(axis="y", labelsize=8)

        ax.set_xticks(np.arange(-0.5, cm_left.shape[1], 1), minor=True)
        ax.set_yticks(np.arange(-0.5, cm_left.shape[0], 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=0.8)
        ax.tick_params(which="minor", bottom=False, left=False)

    plt.tight_layout()
    if save_path is not None:
        save_figure(fig, save_path, dpi=300)
    if show:
        plt.show()

    return fig, axes