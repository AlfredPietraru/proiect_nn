from __future__ import annotations

import numpy as np
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import cm as mpl_cm
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.colors import Normalize

sns.set_theme(style="whitegrid")


def tsne_embeddings(
    X: np.ndarray, perplexity: float = 30.0,
    n_iter: int = 3000, seed: int = 42
) -> np.ndarray:
    X = np.asarray(X)
    n = X.ndim
    if n < 2 or perplexity >= X.shape[0] / 3:
        raise ValueError(
            f"t-SNE requires at least 2D data and "
            f"perplexity < num_samples / 3 (got {X.shape[0]} samples, "
            f"perplexity={perplexity}).")
    tsne = TSNE(
        n_components=2, perplexity=perplexity, learning_rate="auto",
        init="pca", random_state=seed, max_iter=n_iter, n_jobs=-1)
    Z = tsne.fit_transform(X.astype(np.float32, copy=False))
    return Z


def plot_tsne_labels(
    sup_train_2d: np.ndarray, sup_test_2d: np.ndarray,
    sup_train_labels: np.ndarray, sup_test_labels: np.ndarray,
    class_names: list[str], num_classes: int | None = None,
    s_train: int = 40, s_test: int = 60, figsize: tuple[int, int] = (10, 8),
    title: str = "Supervised CNN Repr (t-SNE)\n(train=o, test=x)",
    show: bool = True, save_path: str | None = None
) -> tuple[Figure, Axes]:
    sup_train_2d = np.asarray(sup_train_2d)
    sup_test_2d = np.asarray(sup_test_2d)
    sup_train_labels = np.asarray(sup_train_labels).astype(np.int64, copy=False)
    sup_test_labels = np.asarray(sup_test_labels).astype(np.int64, copy=False)

    if (sup_train_2d.ndim != 2 or sup_train_2d.shape[1] != 2 or
        sup_test_2d.ndim != 2 or sup_test_2d.shape[1] != 2 or
        sup_train_labels.shape[0] != sup_train_2d.shape[0] or
        sup_test_labels.shape[0] != sup_test_2d.shape[0]):
        raise ValueError("Input arrays have incompatible shapes.")

    if num_classes is None:
        max_tr = int(sup_train_labels.max(initial=-1))
        max_te = int(sup_test_labels.max(initial=-1))
        num_classes = max(max_tr, max_te) + 1

    cmap = mpl_cm.get_cmap("tab10" if num_classes <= 10 else "tab20", num_classes)
    palette = [cmap(k) for k in range(num_classes)]
    fig, ax = plt.subplots(figsize=figsize)

    sns.scatterplot(
        x=sup_train_2d[:, 0], y=sup_train_2d[:, 1],
        hue=sup_train_labels.astype(int),
        palette=palette, s=s_train, alpha=0.9, marker="o",
        ax=ax, legend=False)
    sns.scatterplot(
        x=sup_test_2d[:, 0], y=sup_test_2d[:, 1],
        hue=sup_test_labels.astype(int),
        palette=palette, s=s_test, alpha=0.8, marker="x",
        ax=ax, legend=False)

    used = sorted(set(sup_train_labels.tolist()) | set(sup_test_labels.tolist()))
    used = [int(k) for k in used if 0 <= int(k) < num_classes]
    handles = [
        Line2D(
            [0], [0], marker="o", linestyle="",
            markerfacecolor=palette[k], markeredgecolor=palette[k],
            markersize=8, label=(class_names[k] if k < len(class_names) else f"Class {k}"))
        for k in used
    ]

    ax.set_title(title, fontsize=13)
    ax.set_xlabel("t-SNE X")
    ax.set_ylabel("t-SNE y")
    ax.grid(True, alpha=0.2)
    if handles:
        ax.legend(handles=handles, loc="best", fontsize=8, frameon=True, ncol=2)

    norm = Normalize(vmin=0, vmax=num_classes - 1)
    sm = mpl_cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    if num_classes > 1:
        cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Class id")
        cbar.set_ticks([float(i) for i in range(num_classes)])
    else:
        ax.text(0.95, 0.0, f"Class id: {class_names[0] if class_names else '0'}", 
                transform=ax.transAxes, ha="right", va="bottom", fontsize=9)

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()

    return fig, ax


def plot_tsne_transf(
    train_2d: np.ndarray, test_2d: np.ndarray,
    train_labels: np.ndarray, test_labels: np.ndarray,
    class_names: list[str], num_classes: int | None = None,
    figsize: tuple[int, int] = (14, 6), s: int = 50, alpha: float = 0.7,
    title_train: str = "Semi-Supervised Repr (Train)",
    title_test: str = "Semi-Supervised Repr (Test)",
    show: bool = True, save_path: str | None = None
) -> tuple[Figure, np.ndarray]:
    train_2d = np.asarray(train_2d)
    test_2d = np.asarray(test_2d)
    train_labels = np.asarray(train_labels)
    test_labels = np.asarray(test_labels)

    if (train_2d.ndim != 2 or train_2d.shape[1] != 2 or
        test_2d.ndim != 2 or test_2d.shape[1] != 2 or
        train_labels.shape[0] != train_2d.shape[0] or
        test_labels.shape[0] != test_2d.shape[0]):
        raise ValueError("Input arrays have incompatible shapes.")

    if not np.issubdtype(train_labels.dtype, np.number) or not np.issubdtype(test_labels.dtype, np.number):
        uniq = [u for u in np.unique(train_labels) if u is not None]
        label_map = {u: i for i, u in enumerate(uniq)}
        train_labels = np.array([label_map.get(v, -1) for v in train_labels], dtype=np.int64)
        test_labels = np.array([label_map.get(v, -1) for v in test_labels], dtype=np.int64)

    if num_classes is None:
        max_tr = int(train_labels[train_labels >= 0].max(initial=-1))
        max_te = int(test_labels[test_labels >= 0].max(initial=-1))
        num_classes = max(max_tr, max_te) + 1

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    cmap_obj = mpl_cm.get_cmap("tab" + ("10" if num_classes <= 10 else "20"), num_classes)
    palette = [cmap_obj(k) for k in range(num_classes)]

    sns.scatterplot(
        x=train_2d[:, 0], y=train_2d[:, 1],
        hue=train_labels.astype(int),
        palette=palette, s=s, alpha=alpha,
        ax=axes[0], legend=False)
    axes[0].set_title(title_train, fontsize=12)
    axes[0].set_xlabel("t-SNE X")
    axes[0].set_ylabel("t-SNE y")
    axes[0].grid(True, alpha=0.2)

    sns.scatterplot(
        x=test_2d[:, 0], y=test_2d[:, 1],
        hue=test_labels.astype(int),
        palette=palette, s=s, alpha=alpha,
        ax=axes[1], legend=False)
    axes[1].set_title(title_test, fontsize=12)
    axes[1].set_xlabel("t-SNE X")
    axes[1].set_ylabel("t-SNE y")
    axes[1].grid(True, alpha=0.2)

    used = sorted(set(train_labels.tolist()) | set(test_labels.tolist()))
    used = [int(k) for k in used if 0 <= int(k) < num_classes]
    handles = [
        Line2D(
            [0], [0], marker="o", linestyle="",
            markerfacecolor=palette[k], markeredgecolor=palette[k],
            markersize=8, label=(class_names[k] if k < len(class_names) else f"Class {k}"))
        for k in used
    ]
    if handles:
        axes[1].legend(handles=handles, loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True, fontsize=9)

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()

    return fig, axes
