from __future__ import annotations

from typing import Mapping, Tuple, Any, cast

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from data.datasets.config import ClassInfo


def to_numpy_boxes(boxes: Any) -> np.ndarray:
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


def get_info(classes: Mapping[int, ClassInfo], cid: int) -> ClassInfo:
    info = classes.get(int(cid))
    if info is None:
        return ClassInfo(name=f"{int(cid)}", color="#FFFFFF")
    return info


def legend_handles(classes: Mapping[int, ClassInfo], used: set[int]) -> list[Patch]:
    handles: list[Patch] = []
    for cid in sorted(used):
        info = get_info(classes, cid)
        handles.append(Patch(
            label=info.name,
            facecolor=info.color, edgecolor=info.color,
            alpha=0.8, linewidth=2.0))
    return handles


def draw_bbox(
    image: np.ndarray,
    boxes: Any, labels: Any, scores: Any = None, *,
    classes: Mapping[int, ClassInfo],
    conf_thr: float = 0.5, ax: Axes | None = None,
    title: str = "BBoxes", figsize: tuple[int, int] = (12, 8),
) -> Tuple[Figure, Axes, set[int]]:
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = cast(Figure, ax.get_figure())

    img = image
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 1)

    ax.imshow(img)
    ax.axis("off")

    H, W = image.shape[:2]
    boxes_np = to_numpy_boxes(boxes)
    labels_l = as_list(labels)
    scores_l = as_list(scores) if scores is not None else None

    if len(labels_l) != len(boxes_np):
        raise ValueError("Labels length must match number of boxes.")
    if scores_l is not None and len(scores_l) != len(boxes_np):
        raise ValueError("Scores length must match number of boxes.")

    used: set[int] = set()

    for i, (box, lab) in enumerate(zip(boxes_np, labels_l)):
        if scores_l is not None and float(scores_l[i]) < conf_thr:
            continue

        cid = int(lab)
        info = get_info(classes, cid)

        x1, y1, x2, y2 = [float(v) for v in box.tolist()]
        w, h = x2 - x1, y2 - y1
        if w <= 0 or h <= 0:
            continue

        x1 = max(0.0, min(W - 1.0, x1))
        y1 = max(0.0, min(H - 1.0, y1))
        w = max(1.0, min(W - x1, w))
        h = max(1.0, min(H - y1, h))

        rect = Rectangle((x1, y1), w, h, fill=False, linewidth=2.2, edgecolor=info.color, alpha=0.9)
        ax.add_patch(rect)

        score_txt = f" {float(scores_l[i]):.2f}" if scores_l is not None else ""
        label_txt = f"{info.name}{score_txt}"
        ax.text(
            x1 + 2, max(2, y1 - 3),
            label_txt, fontsize=9, color="white", weight="bold", va="top",
            bbox=dict(facecolor=info.color, alpha=0.85, edgecolor=info.color, pad=2))

        used.add(cid)

    if used:
        ax.legend(handles=legend_handles(classes, used), loc="upper right", fontsize=9, framealpha=0.9)

    ax.set_title(title, fontsize=14, fontweight="bold", pad=10)
    plt.tight_layout()
    return fig, ax, used


def show_bbox(
    images: list[np.ndarray],
    boxes: Any, labels: Any, scores: Any = None, *,
    classes: Mapping[int, ClassInfo],
    titles: list[str] | None = None, cols: int = 4
) -> tuple[Figure, np.ndarray]:
    n = len(images)
    rows = (n + cols - 1) // cols
    fig, axes_grid = plt.subplots(rows, cols, figsize=(cols * 6, rows * 4))
    axes = np.atleast_1d(axes_grid).ravel()

    boxes_list = list(boxes)
    labels_list = list(labels)
    scores_list = list(scores) if scores is not None else None

    last = -1
    used_all: set[int] = set()

    for i in range(min(n, len(axes))):
        sc = scores_list[i] if scores_list is not None else None
        title = titles[i] if titles is not None else f"Image {i + 1}"

        _, _, used = draw_bbox(
            image=images[i], boxes=boxes_list[i], labels=labels_list[i],
            scores=sc, classes=classes, ax=axes[i], title=title, figsize=(6, 4))
        used_all.update(used)
        last = i

    for j in range(last + 1, len(axes)):
        axes[j].axis("off")

    fig.suptitle(f"BBoxes ({n} image(s))", fontsize=18, fontweight="bold", y=0.98)
    plt.tight_layout()
    return fig, axes


def save_bbox_visualization(
    image: np.ndarray,
    boxes: Any, labels: Any, scores: Any = None, *,
    classes: Mapping[int, ClassInfo],
    out_pth: str = "bboxes.png", dpi: int = 300, **kwargs
) -> None:
    fig, _, _ = draw_bbox(image, boxes, labels, scores=scores, classes=classes, **kwargs)
    fig.savefig(out_pth, dpi=dpi, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
