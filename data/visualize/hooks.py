from __future__ import annotations

from typing import Dict, List
import torch
import numpy as np

from data.visualize.agreement import agreement_matrix, plot_agreement_heatmap
from data.visualize.plot_kl import plot_kl_stagewise
from data.visualize.tsne import plot_tsne_labels


@torch.no_grad()
def plot_teacher_student_agreement(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    class_names: List[str],
    title: str,
):
    t = teacher_logits.argmax(dim=1).cpu().numpy()
    s = student_logits.argmax(dim=1).cpu().numpy()
    cm = agreement_matrix(t, s, num_classes=len(class_names))
    plot_agreement_heatmap(cm, class_names, title=title)


def plot_tsne(
    sup_train_2d: np.ndarray, sup_test_2d: np.ndarray,
    sup_train_labels: np.ndarray, sup_test_labels: np.ndarray,
    class_names: list[str], num_classes: int | None = None,
    **kwargs,
):
    return plot_tsne_labels(
        sup_train_2d=sup_train_2d,
        sup_test_2d=sup_test_2d,
        sup_train_labels=sup_train_labels,
        sup_test_labels=sup_test_labels,
        class_names=class_names,
        num_classes=num_classes,
        **kwargs,
    )


def plot_stagewise_kl(
    kl_hist: Dict[str, List[float]],
    arch_name: str,
):
    plot_kl_stagewise(
        kl_ema=kl_hist.get("ema", []),
        kl_teacher=kl_hist.get("teacher", []),
        kl_kdd=kl_hist.get("kdd", []),
        arch_name=arch_name,
    )
