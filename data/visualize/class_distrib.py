from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from data.visualize.visualize_common import save_figure
from data.datasets.config import VOC_CLASSES, UAVDT_CLASSES, VISDRONE_CLASSES, AUAIR_CLASSES


def is_string_labels(y: np.ndarray) -> bool:
    """Labels must be strings (unicode, bytes, or object)."""
    return np.asarray(y).dtype.kind in {"U", "S", "O"}


def map_string_labels_to_ids(
    y_train: np.ndarray, y_val: np.ndarray,
    class_names: Sequence[str]
) -> Tuple[np.ndarray, np.ndarray]:
    """Map string labels to integer IDs based on provided class names."""
    name_to_id = {str(name): i for i, name in enumerate(class_names)}
    try:
        y_train_ids = np.asarray([name_to_id[str(v)] for v in y_train], dtype=np.int64)
        y_val_ids = np.asarray([name_to_id[str(v)] for v in y_val], dtype=np.int64)
    except KeyError as e:
        raise ValueError(f"Label not found in class names: {e}") from e
    return y_train_ids, y_val_ids


def sparse_xticks(num_classes: int, max_xticks: int) -> np.ndarray:
    """Generate sparse x-tick indices for plotting."""
    if num_classes <= max_xticks:
        return np.arange(num_classes)
    step = int(np.ceil(num_classes / max_xticks))
    return np.arange(0, num_classes, step)


def build_tick_labels(
    num_classes: int,
    class_names: Optional[Sequence[str]] = None
) -> Sequence[str]:
    """
    Build x-tick labels for plotting.

    If no. classes exceeds the provided class names, fill remaining with
    stringified indices.
    """
    if class_names is None:
        return [str(i) for i in range(num_classes)]
    tick_labels = [str(n) for n in class_names]
    if len(tick_labels) < num_classes:
        tick_labels += [str(i) for i in range(len(tick_labels), num_classes)]
    return tick_labels[:num_classes]


def plot_class_distribution(
    y_train: np.ndarray, y_val: np.ndarray,
    class_names: Optional[Sequence[str]] = None,
    figsize: Tuple[int, int] = (18, 5),
    max_xticks: int = 20,
    rotate: int = 60,
    use_config_colors: bool = True,
    show: bool = True, save_path: Optional[str] = None
) -> Tuple[Figure, np.ndarray]:
    """
    Plot class distribution for training and validation/test sets.

    Produces two side-by-side bar charts:
    - left: train label counts
    - right: validation/test label counts

    If labels are strings/objects, class_names is required to map names -> IDs.

    If use_config_colors=True and class_names provided, bars use colors from config.py
    (VOC/UAVDT/VisDrone/AU-AIR ClassInfo.color).
    """
    y_train = np.asarray(y_train)
    y_val = np.asarray(y_val)

    if is_string_labels(y_train) or is_string_labels(y_val):
        if class_names is None:
            raise ValueError("String labels detected, but must be provided class names.")
        y_train, y_val = map_string_labels_to_ids(y_train, y_val, class_names)
    else:
        y_train = y_train.astype(np.int64, copy=False)
        y_val = y_val.astype(np.int64, copy=False)

    max_train = int(y_train.max(initial=-1))
    max_val = int(y_val.max(initial=-1))
    num_classes = max(max_train, max_val) + 1
    if num_classes <= 0:
        raise ValueError("No classes found (empty labels?).")

    train_counts = np.bincount(y_train, minlength=num_classes)
    val_counts = np.bincount(y_val, minlength=num_classes)
    x = np.arange(num_classes)

    tick_labels = build_tick_labels(num_classes, class_names)
    tick_idx = sparse_xticks(num_classes, max_xticks)

    bar_colors = None
    if use_config_colors and class_names is not None:
        name_to_color = {}
        for m in (VOC_CLASSES, UAVDT_CLASSES, VISDRONE_CLASSES, AUAIR_CLASSES):
            for _, info in m.items():
                name_to_color[str(info.name)] = str(info.color)

        bar_colors = [name_to_color.get(str(n), None) for n in tick_labels]

    fig, axes = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)

    axes[0].bar(x, train_counts, color=bar_colors, edgecolor="black")
    axes[0].set_title("Train Sample Distribution")
    axes[0].set_xlabel("Class")
    axes[0].set_ylabel("Count")
    axes[0].set_xticks(tick_idx)
    axes[0].set_xticklabels([tick_labels[i] for i in tick_idx], rotation=rotate, ha="right")

    axes[1].bar(x, val_counts, color=bar_colors, edgecolor="black")
    axes[1].set_title("Validation/Test Sample Distribution")
    axes[1].set_xlabel("Class")
    axes[1].set_ylabel("Count")
    axes[1].set_xticks(tick_idx)
    axes[1].set_xticklabels([tick_labels[i] for i in tick_idx], rotation=rotate, ha="right")

    if save_path is not None:
        save_figure(fig, save_path)
    if show:
        plt.show()

    return fig, axes
