from __future__ import annotations

from typing import Any, Optional, Sequence, Mapping

import os
import numpy as np
import torch
from matplotlib.figure import Figure

from data.datasets.config import ClassInfo


def normalize_rows(x: np.ndarray) -> np.ndarray:
    """
    Row-normalize a 2D matrix so each row sums to 1.0.
    Useful for confusion/agreement matrices.
    """
    x = np.asarray(x, dtype=float)
    rs = x.sum(axis=1, keepdims=True)
    rs[rs == 0] = 1.0
    return x / rs


def wrap_text(s: str, wrap: int) -> str:
    """Insert newlines into long strings to keep titles readable."""
    if wrap is None or wrap <= 0 or len(s) <= wrap:
        return s
    return "\n".join(s[i:i + wrap] for i in range(0, len(s), wrap))


def format_series(
    x: float | Sequence[float] | None,
    decimals: int = 4,
    max_items: int | None = None,
) -> str | None:
    """Format a float or a short list/array of floats (used for acc/loss legends)."""
    if x is None:
        return None

    if isinstance(x, (list, tuple, np.ndarray)):
        arr = np.asarray(x, dtype=float).ravel()
        suffix = ""
        if max_items is not None and len(arr) > max_items:
            arr = arr[:max_items]
            suffix = ", ..."
        return ", ".join(f"{v:.{decimals}f}" for v in arr.tolist()) + suffix

    if isinstance(x, float):
        return f"{x:.{decimals}f}"

    return str(x)


def ensure_parent_dir(path: str) -> None:
    """Create parent directory if needed (no-op for empty parents)."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def save_figure(fig: Figure, path: str, dpi: int = 300) -> None:
    """Save figure with parent dir creation and tight bbox."""
    ensure_parent_dir(path)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")


def to_numpy_image(
    img: torch.Tensor,
    mean: Optional[Sequence[float]] = None,
    std: Optional[Sequence[float]] = None,
) -> np.ndarray:
    """
    Convert CHW torch image to HWC numpy in [0,1].
    If mean/std are given, unnormalize first.
    """
    x = img.detach().cpu().float()

    if mean is not None and std is not None:
        mean_t = torch.tensor(mean, dtype=torch.float32).view(-1, 1, 1)
        std_t = torch.tensor(std, dtype=torch.float32).view(-1, 1, 1)
        x = x * std_t + mean_t

    x = x.clamp(0, 1).numpy()
    return np.transpose(x, (1, 2, 0))


def to_numpy_boxes_xyxy(boxes: Any) -> np.ndarray:
    """
    Convert boxes to a numpy array of shape (N,4) in XYXY.
    Supports: BoxList-like (.boxes), torch tensor, numpy array, or None.
    """
    if boxes is None:
        return np.zeros((0, 4), dtype=np.float32)

    if hasattr(boxes, "boxes"):
        boxes = boxes.boxes

    if torch.is_tensor(boxes):
        boxes = boxes.detach().cpu().numpy()

    arr = np.asarray(boxes, dtype=np.float32)
    if arr.size == 0:
        return np.zeros((0, 4), dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != 4:
        raise ValueError("Boxes must have shape (N, 4).")
    return arr


def as_list(x: Any) -> list[Any]:
    """Convert tensor/ndarray/scalar/iterable to a Python list."""
    if x is None:
        return []
    if torch.is_tensor(x):
        v = x.detach().cpu().tolist()
        return v if isinstance(v, list) else [v]
    if isinstance(x, np.ndarray):
        v = x.tolist()
        return v if isinstance(v, list) else [v]
    return list(x)


def get_info(classes: Mapping[int, ClassInfo], cid: int) -> ClassInfo:
    """Get class info by ID, or return a default if not found."""
    info = classes.get(int(cid))
    if info is None:
        return ClassInfo(name=f"{int(cid)}", color="#FFFFFF")
    return info
