from __future__ import annotations

from typing import Sequence, Tuple, Dict

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.figure import Figure
from matplotlib.axes import Axes


def normalize_rows(x: np.ndarray) -> np.ndarray:
    x = x.astype(float, copy=False)
    rs = x.sum(axis=1, keepdims=True)
    rs[rs == 0] = 1.0
    return x / rs


def agreement_matrix(
    teacher_labels: np.ndarray,
    student_labels: np.ndarray,
    num_classes: int,
    normalize: bool = True,
) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.float32)

    for t, s in zip(teacher_labels, student_labels):
        if 0 <= t < num_classes and 0 <= s < num_classes:
            cm[int(t), int(s)] += 1.0

    return normalize_rows(cm) if normalize else cm


def plot_agreement_heatmap(
    cm: np.ndarray,
    class_names: Sequence[str], *,
    title: str,
    cmap: str = "Blues",
    figsize: Tuple[int, int] = (8, 6),
    show: bool = True,
) -> Tuple[Figure, Axes]:

    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        cm, ax=ax, annot=True, fmt=".2f",
        cmap=cmap, square=True, cbar=True,
        xticklabels=class_names,
        yticklabels=class_names,
        linewidths=0.5, linecolor="white",
    )

    ax.set_title(title)
    ax.set_xlabel("Student prediction")
    ax.set_ylabel("Teacher prediction")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=60, ha="right")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    plt.tight_layout()
    if show:
        plt.show()

    return fig, ax


def plot_agreement_ema_vs_kdd(
    teacher_ema: np.ndarray, student_ema: np.ndarray,
    teacher_kdd: np.ndarray, student_kdd: np.ndarray,
    class_names: Sequence[str], *,
    arch_name: str,
    figsize: Tuple[int, int] = (16, 6),
    show: bool = True,
) -> Tuple[Figure, np.ndarray]:

    num_classes = len(class_names)
    cm_ema = agreement_matrix(teacher_ema, student_ema, num_classes)
    cm_kdd = agreement_matrix(teacher_kdd, student_kdd, num_classes)

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    sns.heatmap(cm_ema, ax=axes[0], annot=True, fmt=".2f",
                cmap="Blues", square=True,
                xticklabels=class_names, yticklabels=class_names)
    axes[0].set_title(f"{arch_name} - EMA")

    sns.heatmap(cm_kdd, ax=axes[1], annot=True, fmt=".2f",
                cmap="Greens", square=True,
                xticklabels=class_names, yticklabels=class_names)
    axes[1].set_title(f"{arch_name} - KDD")

    for ax in axes:
        ax.set_xlabel("Student")
        ax.set_ylabel("Teacher")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=60, ha="right")
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    plt.tight_layout()
    if show:
        plt.show()

    return fig, axes


def plot_cross_arch_agreement(
    agreement_by_arch: Dict[str, np.ndarray],
    class_names: Sequence[str], *,
    teacher_arch: str,
    figsize: Tuple[int, int] = (18, 6),
    show: bool = True,
) -> Tuple[Figure, np.ndarray]:

    fig, axes = plt.subplots(1, len(agreement_by_arch), figsize=figsize)
    axes = np.atleast_1d(axes)

    for ax, (student_arch, cm) in zip(axes, agreement_by_arch.items()):
        sns.heatmap(
            cm, ax=ax, annot=True, fmt=".2f",
            cmap="Purples", square=True,
            xticklabels=class_names,
            yticklabels=class_names,
        )
        ax.set_title(f"{teacher_arch} - {student_arch}")
        ax.set_xlabel("Student")
        ax.set_ylabel("Teacher")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=60, ha="right")
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    plt.tight_layout()
    if show:
        plt.show()

    return fig, axes
