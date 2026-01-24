from __future__ import annotations

from typing import Any, cast
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import seaborn as sns

from data.datasets.config import ClassInfo
from bbox.box_ops import BoxList
from data.visualize.visualize_common import (
    to_numpy_image, to_numpy_boxes_xyxy,
    as_list, get_info, save_figure)

sns.set_theme(style="whitegrid")


def legend_handles(classes: dict[int, ClassInfo], used: set[int]) -> list[Patch]:
    """Create legend handles for used class IDs."""
    handles: list[Patch] = []
    for cid in sorted(used):
        info = get_info(classes, int(cid))
        handles.append(Patch(
            label=info.name,
            facecolor=info.color, edgecolor=info.color,
            alpha=0.8, linewidth=2.0))
    return handles


def draw_boxes_on_ax(
    ax: Axes, H: int, W: int,
    boxes: Any, labels: Any | None, scores: Any | None,
    classes: dict[int, ClassInfo] | None = None,
    conf_thr: float = 0.0
) -> set[int]:
    """Draw (xyxy) boxes on an existing axis. Returns the set of used class IDs."""
    boxes_np = to_numpy_boxes_xyxy(boxes)
    labels_l = as_list(labels) if labels is not None else [0] * len(boxes_np)
    scores_l = as_list(scores) if scores is not None else None

    if len(labels_l) != len(boxes_np):
        raise ValueError("Labels length must match number of boxes.")
    if scores_l is not None and len(scores_l) != len(boxes_np):
        raise ValueError("Scores length must match number of boxes.")

    used: set[int] = set()
    classes_map = classes or {}

    for i, (box, lab) in enumerate(zip(boxes_np, labels_l)):
        if scores_l is not None and float(scores_l[i]) < conf_thr:
            continue

        cid = int(lab)
        info = get_info(classes_map, cid)

        x1, y1, x2, y2 = [float(v) for v in box.tolist()]
        x1 = max(0.0, min(W - 1.0, x1))
        y1 = max(0.0, min(H - 1.0, y1))
        x2 = max(0.0, min(W - 1.0, x2))
        y2 = max(0.0, min(H - 1.0, y2))
        if x2 <= x1 or y2 <= y1:
            continue

        w = max(1.0, x2 - x1)
        h = max(1.0, y2 - y1)

        ax.add_patch(Rectangle(
            (x1, y1), w, h,
            fill=False, linewidth=2.2,
            edgecolor=info.color, alpha=0.9))

        if labels is not None:
            score_txt = f" {float(scores_l[i]):.2f}" if scores_l is not None else ""
            txt = f"{info.name}{score_txt}"
            ax.text(
                x1 + 2, max(10, y1 + 2), txt,
                fontsize=8, color="white", weight="bold",
                bbox=dict(facecolor=info.color, edgecolor="none", boxstyle="round,pad=0.15", alpha=0.85))

        used.add(cid)

    return used


def draw_bbox(
    image: np.ndarray, boxes: BoxList, labels: list[int],
    scores: list[float] | None, classes: dict[int, ClassInfo] = {},
    conf_thr: float = 0.5, ax: Axes | None = None,
    title: str = "BBoxes", figsize: tuple[int, int] = (12, 8),
    show: bool = True,  save_path: str | None = None
) -> tuple[Figure, Axes, set[int]]:
    """Draw bounding boxes on a single image."""
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = cast(Figure, ax.get_figure())

    img = image
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 1)

    ax.imshow(img)
    ax.axis("off")

    H, W = img.shape[:2]
    used = draw_boxes_on_ax(
        ax, H, W,
        boxes, labels, scores,
        classes=classes,
        conf_thr=conf_thr)

    if used:
        ax.legend(
            handles=legend_handles(classes, used),
            loc="upper right", fontsize=9, framealpha=0.9)

    ax.set_title(title, fontsize=14, fontweight="bold", pad=10)

    plt.tight_layout()
    if save_path is not None:
        save_figure(fig, save_path, dpi=300)
    if show:
        plt.show()

    return fig, ax, used


def set_status_border(ax: Axes, ok: bool | None) -> str:
    """Set the border color of an axis based on prediction status."""
    if ok is None:
        return "black"
    color = "green" if bool(ok) else "red"
    for s in ax.spines.values():
        s.set_color(color)
        s.set_linewidth(3)
    return color


def detect_grid(
    images: torch.Tensor, classes: dict[int, ClassInfo] | None,
    boxes: list[BoxList], labels: list[Any] | None, scores: list[Any] | None,
    mean: list[float] | None, std: list[float] | None, pred_status: list[bool] | None,
    titles: list[str] | str | None,  conf_thr: float = 0.0, grid_title: str = "Detection Grid",
    cols: int = 4, figsize_per_cell: tuple[float, float] = (3.3, 3.3),
    show: bool = True, save_path: str | None = None
) -> tuple[Figure, np.ndarray]:
    """
    Visualize a grid of detection results with bounding boxes.
    Bounding boxes are drawn on each image in the batch.
    """
    B = int(images.size(0))
    if B == 0:
        raise ValueError("Empty batch: images tensor has zero length.")
    if len(boxes) != B:
        raise ValueError(f"Boxes length {len(boxes)} does not match batch size {B}.")

    labels_seq = labels if labels is not None else [None] * B
    scores_seq = scores if scores is not None else [None] * B
    status_seq = pred_status if pred_status is not None else [None] * B
    titles_seq = titles if titles is not None else [f"Image {i}" for i in range(B)]

    ncols = max(1, min(int(cols), B))
    nrows = (B + ncols - 1) // ncols

    fig_w = ncols * float(figsize_per_cell[0])
    fig_h = nrows * float(figsize_per_cell[1])
    fig, axes_grid = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h))
    axes = np.atleast_1d(axes_grid).ravel()

    for i, ax in enumerate(axes):
        ax.axis("off")
        if i >= B:
            continue

        img = to_numpy_image(images[i], mean, std)
        H, W = img.shape[:2]
        ax.imshow(img)

        ok = status_seq[i] if i < len(status_seq) else None
        title_color = set_status_border(ax, ok if ok is None else bool(ok))

        title = titles_seq[i] if i < len(titles_seq) else f"Image {i}"
        ax.set_title(title, fontsize=10, color=title_color)

        used = draw_boxes_on_ax(
            ax, H, W,
            boxes[i], labels_seq[i], scores_seq[i],
            classes=classes, conf_thr=conf_thr)

        if used and classes is not None:
            ax.legend(
                handles=legend_handles(classes, used),
                loc="upper right", fontsize=7, framealpha=0.9)

    for j in range(B, len(axes)):
        axes[j].axis("off")

    fig.suptitle(grid_title, fontsize=16)

    plt.tight_layout()
    if save_path is not None:
        save_figure(fig, save_path, dpi=300)
    if show:
        plt.show()

    return fig, axes
