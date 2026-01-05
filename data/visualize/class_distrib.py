from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure


def plot_class_distribution(
    y_train: np.ndarray,
    y_val: np.ndarray,
    class_names: Optional[Sequence[str]] = None,
    figsize: Tuple[int, int] = (18, 5),
    show: bool = True,
    max_xticks: int = 20,
    rotate: int = 60,
) -> Tuple[Figure, np.ndarray]:
    y_train = np.asarray(y_train)
    y_val = np.asarray(y_val)

    string_labels = (y_train.dtype.kind in {"U", "S", "O"}) or (y_val.dtype.kind in {"U", "S", "O"})
    if string_labels:
        if class_names is None:
            raise ValueError("String labels detected, but must be provided class names.")

        name_to_id = {str(name): i for i, name in enumerate(class_names)}
        try:
            y_train = np.asarray([name_to_id[str(v)] for v in y_train], dtype=np.int64)
            y_val = np.asarray([name_to_id[str(v)] for v in y_val], dtype=np.int64)
        except KeyError as e:
            raise ValueError(f"Label not found in class_names: {e}") from e
    else:
        y_train = y_train.astype(np.int64, copy=False)
        y_val = y_val.astype(np.int64, copy=False)

    # Determine number of classes
    max_train = int(y_train.max(initial=-1))
    max_val = int(y_val.max(initial=-1))
    num_classes = max(max_train, max_val) + 1
    if num_classes <= 0:
        raise ValueError("No classes found (empty labels?).")

    train_counts = np.bincount(y_train, minlength=num_classes)
    val_counts = np.bincount(y_val, minlength=num_classes)
    x = np.arange(num_classes)

    # Tick labels
    if class_names is None:
        tick_labels = [str(i) for i in range(num_classes)]
    else:
        tick_labels = [str(n) for n in class_names]
        if len(tick_labels) < num_classes:
            tick_labels += [str(i) for i in range(len(tick_labels), num_classes)]
        tick_labels = tick_labels[:num_classes]

    # Sparse x-ticks when there are many classes
    if num_classes <= max_xticks:
        tick_idx = x
    else:
        step = int(np.ceil(num_classes / max_xticks))
        tick_idx = np.arange(0, num_classes, step)

    fig, axes = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)

    axes[0].bar(x, train_counts, edgecolor="black")
    axes[0].set_title("Train Sample Distribution")
    axes[0].set_xlabel("Class")
    axes[0].set_ylabel("Count")
    axes[0].set_xticks(tick_idx)
    axes[0].set_xticklabels([tick_labels[i] for i in tick_idx], rotation=rotate, ha="right")

    axes[1].bar(x, val_counts, edgecolor="black")
    axes[1].set_title("Validation/Test Sample Distribution")
    axes[1].set_xlabel("Class")
    axes[1].set_ylabel("Count")
    axes[1].set_xticks(tick_idx)
    axes[1].set_xticklabels([tick_labels[i] for i in tick_idx], rotation=rotate, ha="right")

    if show:
        plt.show()

    return fig, axes
