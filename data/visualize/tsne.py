from __future__ import annotations

from typing import Tuple

import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.colors import Normalize


def plot_tsne_labels(
    sup_train_2d: np.ndarray, sup_test_2d: np.ndarray,
    sup_train_labels: np.ndarray, sup_test_labels: np.ndarray,
    class_names: list[str], num_classes: int | None = None,
    figsize: tuple = (10, 8), s_train: int = 18, s_test: int = 28,
    alpha_train: float = 0.75, alpha_test: float = 0.85,
    title: str = "Supervised CNN Representations (t-SNE)\n(train=o, test=x)",
    legend_fontsize: int = 8, show: bool = True,
) -> Tuple[Figure, Axes]:
    fig, ax = plt.subplots(figsize=figsize)

    sup_train_labels = np.asarray(sup_train_labels)
    sup_test_labels = np.asarray(sup_test_labels)

    if num_classes is None:
        num_classes = int(max(sup_train_labels.max(), sup_test_labels.max()) + 1)

    cmap = cm.get_cmap("tab10" if num_classes <= 10 else "tab20", num_classes)

    for k in range(num_classes):
        tr_mask = (sup_train_labels == k)
        te_mask = (sup_test_labels == k)
        if not tr_mask.any() and not te_mask.any():
            continue

        ax.scatter(
            sup_train_2d[tr_mask, 0], sup_train_2d[tr_mask, 1],
            s=s_train, alpha=alpha_train, marker="o",
            color=cmap(k), label=class_names[k]
        )
        ax.scatter(
            sup_test_2d[te_mask, 0], sup_test_2d[te_mask, 1],
            s=s_test, alpha=alpha_test, marker="x",
            color=cmap(k),
        )

    ax.set_title(title, fontsize=13)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.grid(True, alpha=0.2)
    ax.legend(loc="best", fontsize=legend_fontsize, frameon=True, ncol=2)

    # Colorbar
    norm = Normalize(vmin=0, vmax=num_classes - 1)
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Class id")
    cbar.set_ticks([float(i) for i in range(num_classes)])

    fig.tight_layout()
    if show:
        plt.show()

    return fig, ax


def plot_tsne_transf(
    train_2d: np.ndarray, test_2d: np.ndarray,
    train_labels: np.ndarray, test_labels: np.ndarray,
    class_names: list[str] | None = None,
    num_classes: int | None = None,
    figsize: tuple = (14, 6),
    s: int = 50, alpha: float = 0.7, cmap: str = "tab10",
    title_train: str = "Semi-Supervised Representations (Train)",
    title_test: str = "Semi-Supervised Representations (Test)",
    show: bool = True, save_path: str | None = None, dpi: int = 150,
) -> Tuple[Figure, Axes]:
    train_2d = np.asarray(train_2d)
    test_2d = np.asarray(test_2d)
    train_labels = np.asarray(train_labels)
    test_labels = np.asarray(test_labels)

    # Map non-numeric labels to ids
    if not np.issubdtype(train_labels.dtype, np.number):
        uniq = [u for u in np.unique(train_labels) if u is not None]
        label_map = {u: i for i, u in enumerate(uniq)}
        train_labels = np.array([label_map.get(v, -1) for v in train_labels])
        test_labels = np.array([label_map.get(v, -1) for v in test_labels])

    # Compute num_classes if not provided
    if num_classes is None:
        num_classes = int(max(train_labels.max(), test_labels.max()) + 1)

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    cm = cm.get_cmap(cmap, num_classes)

    # TRAIN (per class for legend)
    for k in range(num_classes):
        m = (train_labels == k)
        if not m.any():
            continue
        axes[0].scatter(
            train_2d[m, 0], train_2d[m, 1],
            color=cm(k), alpha=alpha, s=s,
            label=(class_names[k] if class_names is not None else f"Class {k}")
        )

    axes[0].set_title(title_train, fontsize=12)
    axes[0].set_xlabel("t-SNE 1")
    axes[0].set_ylabel("t-SNE 2")
    axes[0].grid(True, alpha=0.2)

    # TEST
    for k in range(num_classes):
        m = (test_labels == k)
        if not m.any():
            continue
        axes[1].scatter(
            test_2d[m, 0], test_2d[m, 1],
            color=cm(k), alpha=alpha, s=s,
            label=(class_names[k] if class_names is not None else f"Class {k}")
        )

    axes[1].set_title(title_test, fontsize=12)
    axes[1].set_xlabel("t-SNE 1")
    axes[1].set_ylabel("t-SNE 2")
    axes[1].grid(True, alpha=0.2)

    # One shared legend
    axes[1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True, fontsize=9)

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    return fig, axes
