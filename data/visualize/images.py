from __future__ import annotations

import os
from typing import Any, Mapping, Sequence, Tuple, Optional

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from data.datasets.config import ClassInfo


def to_numpy_image(
    img: torch.Tensor,
    mean: Optional[Sequence[float]],
    std: Optional[Sequence[float]]
) -> np.ndarray:
    x = img.detach().cpu().float()

    if mean is not None and std is not None:
        mean_t = torch.tensor(mean, dtype=torch.float32).view(-1, 1, 1)
        std_t = torch.tensor(std, dtype=torch.float32).view(-1, 1, 1)
        x = x * std_t + mean_t

    x = x.clamp(0, 1).numpy()
    return np.transpose(x, (1, 2, 0))


def to_numpy_boxes(boxes: Any) -> np.ndarray:
    if boxes is None:
        return np.zeros((0, 4), dtype=np.float32)
    if torch.is_tensor(boxes):
        boxes = boxes.detach().cpu().numpy()
    arr = np.asarray(boxes, dtype=np.float32)
    if arr.size == 0:
        return np.zeros((0, 4), dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != 4:
        raise ValueError("Boxes must have shape (N, 4).")
    return arr


def as_list(x: Any) -> list[Any]:
    if x is None:
        return []
    # Tensor -> list
    if torch.is_tensor(x):
        res = x.detach().cpu().tolist()
        if not isinstance(res, list):
            return [res]
        return res
    # Numpy array -> list
    if isinstance(x, np.ndarray):
        res = x.tolist()
        if not isinstance(res, list):
            return [res]
        return res
    return list(x)


def get_info(classes: Optional[Mapping[int, ClassInfo]], cid: int) -> ClassInfo:
    if classes is None:
        return ClassInfo(name=str(int(cid)), color="#00BFFF")
    info = classes.get(int(cid))
    if info is None:
        return ClassInfo(name=str(int(cid)), color="#00BFFF")
    return info


def draw_boxes_on_ax(
    ax: Axes, *,
    H: int, W: int,
    boxes: Any, labels: Optional[Any], scores: Optional[Any],
    classes: Optional[Mapping[int, ClassInfo]] = None,
    conf_thr: float = 0.0,
) -> None:
    boxes_np = to_numpy_boxes(boxes)
    labels_l = as_list(labels) if labels is not None else [0] * len(boxes_np)
    scores_l = as_list(scores) if scores is not None else None

    if len(labels_l) != len(boxes_np):
        raise ValueError("Labels length must match number of boxes.")
    if scores_l is not None and len(scores_l) != len(boxes_np):
        raise ValueError("Scores length must match number of boxes.")

    for i, (box, lab) in enumerate(zip(boxes_np, labels_l)):
        if scores_l is not None and float(scores_l[i]) < conf_thr:
            continue

        cid = int(lab)
        info = get_info(classes, cid)

        x1, y1, x2, y2 = [float(v) for v in box.tolist()]
        x1 = max(0.0, min(W - 1.0, x1))
        y1 = max(0.0, min(H - 1.0, y1))
        x2 = max(0.0, min(W - 1.0, x2))
        y2 = max(0.0, min(H - 1.0, y2))

        w = max(1.0, x2 - x1)
        h = max(1.0, y2 - y1)

        ax.add_patch(Rectangle((x1, y1), w, h, fill=False, linewidth=2, edgecolor=info.color, alpha=0.9))

        if labels is not None:
            score_txt = f" {float(scores_l[i]):.2f}" if scores_l is not None else ""
            txt = f"{info.name}{score_txt}"
            ax.text(
                x1 + 2, max(10, y1 + 2),
                txt, fontsize=8, color="white", weight="bold",
                bbox=dict(facecolor=info.color, edgecolor="none", boxstyle="round,pad=0.15", alpha=0.85))


def set_status_border(ax: Axes, ok: bool | None) -> str:
    if ok is None:
        return "black"

    color = "green" if bool(ok) else "red"
    for s in ax.spines.values():
        s.set_color(color)
        s.set_linewidth(3)
    return color


def save_detection_grid(
    images: torch.Tensor,
    titles: Sequence[str],
    boxes: Sequence[Any],
    labels: Optional[Sequence[Any]],
    scores: Optional[Sequence[Any]],
    classes: Optional[Mapping[int, ClassInfo]],
    mean: Optional[Sequence[float]],
    std: Optional[Sequence[float]],
    pred_status: Optional[Sequence[bool]],
    conf_thr: float = 0.0,
    grid_title: str = "Detection Grid",
    out_path: str = "output/grid.png",
    cols: int = 4,
    figsize_per_cell: Tuple[float, float] = (3.3, 3.3),
    dpi: int = 300,
) -> Tuple[Figure, np.ndarray]:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    B = int(images.size(0))
    if B == 0:
        raise ValueError("Empty batch: images tensor has zero length.")
    if len(boxes) != B:
        raise ValueError(f"Boxes length {len(boxes)} does not match batch size {B}.")

    labels_seq = labels if labels is not None else [None] * B
    scores_seq = scores if scores is not None else [None] * B
    status_seq = pred_status if pred_status is not None else [None] * B

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

        img = to_numpy_image(images[i], mean=mean, std=std)
        H, W = img.shape[:2]
        ax.imshow(img)

        ok = status_seq[i] if i < len(status_seq) else None
        title_color = set_status_border(ax, ok if ok is None else bool(ok))

        title = titles[i] if i < len(titles) else f"Image {i}"
        ax.set_title(title, fontsize=10, color=title_color)

        draw_boxes_on_ax(
            ax, H=H, W=W,
            boxes=boxes[i], labels=labels_seq[i], scores=scores_seq[i],
            classes=classes, conf_thr=conf_thr)

    for j in range(B, len(axes)):
        axes[j].axis("off")

    fig.suptitle(grid_title, fontsize=16)
    plt.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return fig, axes
