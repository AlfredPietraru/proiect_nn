from __future__ import annotations

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import seaborn as sns

from data.visualize.visualize_common import normalize_rows, save_figure

sns.set_theme(style="whitegrid")


def agreement_matrix(
    teacher_labels: np.ndarray, student_labels: np.ndarray,
    num_classes: int, normalize: bool = True
) -> np.ndarray:
    """
    Teacher-student agreement matrix.
    Confusion matrix where rows are teacher labels
    and columns are student labels.
    """
    cm = np.zeros((num_classes, num_classes), dtype=np.float32)

    for t, s in zip(teacher_labels, student_labels):
        if 0 <= t < num_classes and 0 <= s < num_classes:
            cm[int(t), int(s)] += 1.0

    return normalize_rows(cm) if normalize else cm


def plot_agreement_heatmap(
    cm: np.ndarray, class_names: list[str],
    title: str, cmap: str = "Blues",
    figsize: tuple[int, int] = (8, 6), 
    show: bool = True, save_path: str | None = None
) -> tuple[Figure, Axes]:
    """
    Plot agreement heatmap from confusion matrix.
    Visualizes how often student predictions agree with teacher predictions.
    The axes are:
    - Y-axis: teacher predictions (rows)
    - X-axis: student predictions (columns)
    """
    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(cm, ax=ax, annot=True, fmt=".2f",
                cmap=cmap, square=True, cbar=True,
                xticklabels=class_names, yticklabels=class_names,
                linewidths=0.5, linecolor="white")

    ax.set_title(title)
    ax.set_xlabel("Student prediction")
    ax.set_ylabel("Teacher prediction")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=60, ha="right")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    plt.tight_layout()
    if save_path is not None:
        save_figure(fig, save_path)
    if show:
        plt.show()

    return fig, ax


def plot_agreement_ema_vs_kdd(
    teacher_ema: np.ndarray, student_ema: np.ndarray,
    teacher_kdd: np.ndarray, student_kdd: np.ndarray,
    class_names: list[str], arch_name: str,
    figsize: tuple[int, int] = (16, 6), 
    show: bool = True, save_path: str | None = None
) -> tuple[Figure, np.ndarray]:
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
    if save_path is not None:
        save_figure(fig, save_path)
    if show:
        plt.show()

    return fig, axes


def plot_cross_arch_agreement(
    agreement_by_arch: dict[str, np.ndarray],
    class_names: list[str], teacher_arch: str,
    figsize: tuple[int, int] = (18, 6), 
    show: bool = True, save_path: str | None = None
) -> tuple[Figure, np.ndarray]:
    """
    Plot cross-architecture agreement heatmaps.
    All heatmaps share the same teacher architecture.
    """
    fig, axes = plt.subplots(1, len(agreement_by_arch), figsize=figsize)
    axes = np.atleast_1d(axes)

    for ax, (student_arch, cm) in zip(axes, agreement_by_arch.items()):
        sns.heatmap(cm, ax=ax, annot=True, fmt=".2f", 
                    cmap="Purples", square=True,
                    xticklabels=class_names, yticklabels=class_names)
        ax.set_title(f"{teacher_arch} - {student_arch}")
        ax.set_xlabel("Student")
        ax.set_ylabel("Teacher")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=60, ha="right")
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    plt.tight_layout()
    if save_path is not None:
        save_figure(fig, save_path)
    if show:
        plt.show()

    return fig, axes


@torch.no_grad()
def plot_agreement_teacher_vs_student(
    teacher_logits: torch.Tensor, student_logits: torch.Tensor,
    class_names: list[str], title: str, 
    show: bool = True, save_path: str | None = None
) -> None:
    """Plot a teacher-vs-student label agreement heatmap (argmax logits -> agreement matrix)."""
    t = teacher_logits.argmax(dim=1).detach().cpu().numpy()
    s = student_logits.argmax(dim=1).detach().cpu().numpy()
    cm = agreement_matrix(t, s, len(class_names))
    plot_agreement_heatmap(cm, class_names, title=title, show=show, save_path=save_path)
