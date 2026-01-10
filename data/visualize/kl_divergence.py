from __future__ import annotations

from typing import Dict, List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.figure import Figure
from matplotlib.axes import Axes

sns.set_theme(style="whitegrid")


EPS: float = 1e-6


def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """KL divergence D_KL(P || Q) between two distributions."""
    p, q = np.clip(p, EPS, None), np.clip(q, EPS, None)
    p, q = p / np.sum(p), q / np.sum(q)
    return float(np.sum(p * np.log(p / q)))


def plot_kl_stagewise(
    kl_ema: List[float], kl_teacher: List[float], kl_kdd: List[float],
    arch_name: str, figsize: Tuple[int, int] = (10, 6), 
    title: str = "KL Divergence (Stage-wise)",
    show: bool = True, save_path: Optional[str] = None
) -> Tuple[Figure, Axes]:
    """Plot KL divergence over training epochs for different models."""
    epochs = range(1, max(len(kl_ema), len(kl_teacher), len(kl_kdd)) + 1)
    fig, ax = plt.subplots(figsize=figsize)

    if kl_ema:
        sns.lineplot(x=list(epochs)[:len(kl_ema)], y=kl_ema, ax=ax, label="Student EMA", linewidth=2)
    if kl_teacher:
        sns.lineplot(x=list(epochs)[:len(kl_teacher)], y=kl_teacher, ax=ax, label="Teacher", linewidth=2)
    if kl_kdd:
        sns.lineplot(x=list(epochs)[:len(kl_kdd)], y=kl_kdd, ax=ax, label="Student KDD", linewidth=2)

    ax.set_title(f"{title} - {arch_name}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("KL Divergence")
    ax.legend()

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()

    return fig, ax


def plot_cross_arch_kl(
    kl_by_arch: Dict[str, List[float]],
    teacher_arch: str,
    figsize: Tuple[int, int] = (10, 6),
    show: bool = True, save_path: Optional[str] = None
) -> Tuple[Figure, Axes]:
    """Plot KL divergence curves for different student architectures."""
    fig, ax = plt.subplots(figsize=figsize)

    for student_arch, kl in kl_by_arch.items():
        ax.plot(range(1, len(kl) + 1), kl, linewidth=2,
                label=f"{teacher_arch} → {student_arch}")

    ax.set_title("Cross-Architecture Knowledge Transfer (KL)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("KL divergence")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if show:
        plt.show()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig, ax


def plot_cross_dataset_kl(
    kl_by_dataset: Dict[str, List[float]], *,
    teacher_dataset: str, teacher_arch: str,
    figsize: Tuple[int, int] = (10, 6),
    show: bool = True, save_path: Optional[str] = None
) -> Tuple[Figure, Axes]:
    """Plot KL divergence curves for different student datasets."""
    fig, ax = plt.subplots(figsize=figsize)

    for student_dataset, kl in kl_by_dataset.items():
        ax.plot(range(1, len(kl) + 1), kl, linewidth=2,
                label=f"{teacher_dataset} → {student_dataset}")

    ax.set_title(f"Cross-Dataset KL Transfer ({teacher_arch})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("KL divergence")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if show:
        plt.show()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig, ax
