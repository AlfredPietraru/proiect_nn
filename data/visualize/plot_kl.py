from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes


def kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-6) -> float:
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    return float(np.sum(p * (np.log(p) - np.log(q))))


def plot_kl_stagewise(
    kl_ema: List[float],
    kl_teacher: List[float],
    kl_kdd: List[float], *,
    arch_name: str,
    figsize: Tuple[int, int] = (9, 5),
    show: bool = True,
) -> Tuple[Figure, Axes]:

    epochs = range(1, max(len(kl_ema), len(kl_teacher), len(kl_kdd)) + 1)
    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(epochs[:len(kl_ema)], kl_ema, label="Student EMA", linewidth=2)
    ax.plot(epochs[:len(kl_teacher)], kl_teacher, label="Teacher", linewidth=2)
    ax.plot(epochs[:len(kl_kdd)], kl_kdd, label="Student KDD", linewidth=2)

    ax.set_title(f"KL Divergence - {arch_name}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("KL")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if show:
        plt.show()

    return fig, ax


def plot_cross_arch_kl(
    kl_by_arch: Dict[str, List[float]], *,
    teacher_arch: str,
    figsize: Tuple[int, int] = (10, 6),
    show: bool = True,
) -> Tuple[Figure, Axes]:

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

    return fig, ax


def plot_cross_dataset_kl(
    kl_by_dataset: Dict[str, List[float]], *,
    teacher_dataset: str,
    teacher_arch: str,
    figsize: Tuple[int, int] = (10, 6),
    show: bool = True,
) -> Tuple[Figure, Axes]:

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

    return fig, ax
