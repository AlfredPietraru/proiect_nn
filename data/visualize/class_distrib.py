from __future__ import annotations

from typing import Tuple, Sequence, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import seaborn as sns

from data.visualize.visualize_common import normalize_rows, wrap_text, format_series, save_figure

sns.set_theme(style="whitegrid")


def plot_heatmap(
    ax: Axes, cm: np.ndarray, class_names: Sequence[str],
    title: str, fmt: str, cmap: str, rotation_x: int, 
    vmin: Optional[float] = None, vmax: Optional[float] = None
) -> None:
    """Plot a confusion-matrix heatmap on a given axis (rows=true, cols=pred)."""
    sns.heatmap(
        cm, annot=True,
        fmt=fmt, cmap=cmap, vmin=vmin, vmax=vmax,
        cbar=True, square=True, linewidths=0.5, linecolor="white",
        xticklabels=class_names, yticklabels=class_names, ax=ax)

    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=rotation_x, ha="right")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)


def format_acc_series(
    acc: Optional[float | Sequence[float]],
    decimals: int = 4, max_items: Optional[int] = None
) -> str | None:
    """Format an accuracy value (float) or a list of values into a short string."""
    if acc is None:
        return None

    if isinstance(acc, (list, tuple, np.ndarray)):
        arr = np.asarray(acc, dtype=float).ravel()
        suffix = ""
        if max_items is not None and len(arr) > max_items:
            arr = arr[:max_items]
            suffix = ", ..."
        parts = [f"{v:.{decimals}f}" for v in arr.tolist()]
        return ", ".join(parts) + suffix

    if isinstance(acc, float):
        return f"{acc:.{decimals}f}"
    return str(acc)


def plot_confusion_matrix(
    cm: np.ndarray, class_names: Sequence[str], 
    normalize: bool = False, cmap: str = "Greens",
    title: str = "Confusion Matrix", figsize: Tuple[int, int] = (8, 6),
    show: bool = True, save_path: Optional[str] = None
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

    fig, ax = plt.subplots(figsize=figsize)

    fmt: str = "d"
    rotation_x: int = 60
    plot_heatmap(ax, cm, class_names, title, fmt, cmap, rotation_x)
    ax.set_aspect("equal")

    ax.tick_params(axis="x", labelsize=8)
    ax.tick_params(axis="y", labelsize=8)
    plt.setp(ax.get_xticklabels(), rotation=rotation_x, ha="right", rotation_mode="anchor")
    plt.setp(ax.get_yticklabels(), rotation=0, ha="right")

    ax.set_xticks(np.arange(-0.5, cm.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, cm.shape[0], 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.8)
    ax.tick_params(which="minor", bottom=False, left=False)

    plt.tight_layout()
    if save_path is not None:
        save_figure(fig, save_path)
    if show:
        plt.show()

    return fig, ax


def plot_confusion_matrices_side_by_side(
    cm_left: np.ndarray, cm_right: np.ndarray, class_names: Sequence[str],
    left_title: str = "Model A", right_title: str = "Model B",
    left_acc: Optional[float | Sequence[float]] = None,
    right_acc: Optional[float | Sequence[float]] = None,
    left_cmap: str = "Blues", right_cmap: str = "Greens",
    figsize: Tuple[int, int] = (16, 7), 
    fmt: str = "d", shared_scale: bool = True, show: bool = True,
    acc_max_items: Optional[int] = None, save_path: Optional[str] = None
) -> tuple[Figure, Sequence[Axes]]:
    """
    Plot two confusion matrices side-by-side for comparison.
    Matrices are row-normalized to show per-true-class prediction probabilities.
    """
    cm_left = np.asarray(cm_left)
    cm_right = np.asarray(cm_right)

    cm_left = normalize_rows(cm_left)
    cm_right = normalize_rows(cm_right)
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

    plot_heatmap(
        axes[0], cm_left, class_names, left_title, fmt,
        left_cmap, 45, vmin=vmin, vmax=vmax)
    plot_heatmap(
        axes[1], cm_right, class_names, right_title, fmt,
        right_cmap, 45, vmin=vmin, vmax=vmax)

    for ax in axes:
        ax.set_aspect("equal")
        ax.tick_params(axis="x", labelsize=8)
        ax.tick_params(axis="y", labelsize=8)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        ax.set_xticks(np.arange(-0.5, cm_left.shape[1], 1), minor=True)
        ax.set_yticks(np.arange(-0.5, cm_left.shape[0], 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=0.8)
        ax.tick_params(which="minor", bottom=False, left=False)

    plt.tight_layout()
    if save_path is not None:
        save_figure(fig, save_path)
    if show:
        plt.show()

    return fig, axes
