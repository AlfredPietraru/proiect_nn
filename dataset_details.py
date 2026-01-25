from __future__ import annotations

import os
from typing import Dict, List, Tuple, Mapping

import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt

from data.datasets.config import ClassInfo
from data.visualize import (
    plot_class_distribution, tsne_embeddings,
    plot_tsne_labels, plot_tsne_transf)


def extract_class_names(classes: Mapping[int, ClassInfo]) -> List[str]:
    """Extract class names from the classes mapping."""
    ids = sorted(classes.keys())
    return [classes[i].name for i in ids]


def collect_detection_labels(
    loader: DataLoader,
    max_batches: int = 9999
) -> np.ndarray:
    """"Collect all object detection labels from the loader."""
    labels: List[int] = []
    for bidx, (_, targets) in enumerate(loader):
        if bidx >= max_batches:
            break

        for t in targets: 
            y = t.get("labels", None)
            if y is None:
                continue
            if isinstance(y, torch.Tensor):
                y = y.detach().cpu().long().view(-1).tolist()
            else:
                y = list(y)
            labels.extend([int(v) for v in y])

    return np.asarray(labels, dtype=np.int64)


def collect_image_level_labels(
    loader: DataLoader, 
    max_batches: int, max_images: int,
    empty_label: int = -1
) -> np.ndarray:
    """
    For t-SNE coloring we need ONE label per image.
    We take the first object label if present otherwise empty label.
    """
    labels: List[int] = []
    for bidx, (_, targets_batch) in enumerate(loader):
        if bidx >= max_batches or len(labels) >= max_images:
            break
        for t in targets_batch:
            if len(labels) >= max_images:
                break
            y = t.get("labels", None)
            if y is None:
                labels.append(empty_label); continue
            if isinstance(y, torch.Tensor):
                y = y.detach().cpu().long().view(-1)
            else:
                y = torch.tensor(y, dtype=torch.long)
            labels.append(int(y[0].item()) if y.numel() else empty_label)
    return np.asarray(labels, dtype=np.int64)


def images_to_embeddings(
    images: List[torch.Tensor],
    embed_hw: Tuple[int, int]
) -> np.ndarray:
    """
    Simple deterministic embedding: downsample -> flatten.
    images: list of (C,H,W) tensors.
    """
    embs: List[np.ndarray] = []

    for img in images:
        x = img.detach().float().cpu()
        if x.ndim != 3:
            continue

        x = x.unsqueeze(0)  # 1,C,H,W
        x = F.interpolate(x, size=embed_hw, mode="bilinear", align_corners=False)
        x = x.squeeze(0).reshape(-1).numpy()
        embs.append(x)

    if len(embs) == 0:
        return np.zeros((0, int(embed_hw[0] * embed_hw[1] * 3)), dtype=np.float32)

    X = np.stack(embs, axis=0).astype(np.float32, copy=False)
    return X


def collect_image_embeddings(
    loader: DataLoader,
    max_batches: int, max_images: int,
    embed_hw: Tuple[int, int]
) -> np.ndarray:
    """Collect image embeddings by downsampling and flattening images from the loader."""
    all_imgs: List[torch.Tensor] = []

    for bidx, (images, _) in enumerate(loader):
        if bidx >= max_batches:
            break

        for img in images:
            all_imgs.append(img)
            if len(all_imgs) >= int(max_images):
                break

        if len(all_imgs) >= int(max_images):
            break

    return images_to_embeddings(all_imgs, embed_hw)


def dataset_details(
    data: Dict[str, DataLoader],
    classes: Mapping[int, ClassInfo],
    max_batches: int = 50, max_images: int = 400, 
    embed_hw: Tuple[int, int] = (32, 32),
    tsne_perplexity: float = 30.0, tsne_iter: int = 2000,
    seed: int = 42, show: bool = False, save_path: str = "images/details"
) -> None:
    train_loader = data.get("train_burn_in_strong", None)
    test_loader = data.get("test", None)
    if train_loader is None or test_loader is None:
        raise ValueError("Expected the supervised train and test loaders in data.")

    os.makedirs(save_path, exist_ok=True)
    class_names = extract_class_names(classes)

    y_train_obj = collect_detection_labels(train_loader)
    y_test_obj = collect_detection_labels(test_loader)

    if y_train_obj.size > 0 or y_test_obj.size > 0:
        plot_class_distribution(
            y_train_obj, y_test_obj, class_names, 
            show=show, max_xticks=20, rotate=60, save_path=os.path.join(save_path, "burn_in_class_distribution"))
        plt_close(fig=True)

    X_train = collect_image_embeddings(train_loader, max_batches, max_images, embed_hw)
    X_test = collect_image_embeddings(test_loader, max_batches, max_images, embed_hw)

    if X_train.shape[0] >= 10 and X_test.shape[0] >= 10:
        # One label per image for coloring (first object label per image)
        y_train_img = collect_image_level_labels(train_loader, max_batches, max_images, empty_label=-1)
        y_test_img  = collect_image_level_labels(test_loader,  max_batches, max_images, empty_label=-1)

        # Trim labels to match embeddings count
        y_train_img = y_train_img[: X_train.shape[0]]
        y_test_img = y_test_img[: X_test.shape[0]]

        print("X_train:", X_train.shape, "X_test:", X_test.shape)
        print("y_train_img:", y_train_img.shape, "unique:", np.unique(y_train_img, return_counts=True))
        print("y_test_img:", y_test_img.shape, "unique:", np.unique(y_test_img, return_counts=True))

        # TSNE needs enough samples and perplexity < n/3
        p_train = min(float(tsne_perplexity), max(2.0, (X_train.shape[0] / 3.0) - 1.0))
        p_test = min(float(tsne_perplexity), max(2.0, (X_test.shape[0] / 3.0) - 1.0))

        Z_train = tsne_embeddings(X_train, p_train, tsne_iter, seed)
        Z_test = tsne_embeddings(X_test, p_test, tsne_iter, seed)

        # Separate plot (train vs test)
        save_labels = os.path.join(save_path, "dataset_tsne_labels.png") if save_path is not None else None
        save_transf = os.path.join(save_path, "dataset_tsne_transf.png") if save_path is not None else None
        plot_tsne_labels(
            Z_train, Z_test, y_train_img, y_test_img,
            class_names=class_names, show=show, save_path=save_labels)
        plot_tsne_transf(
            Z_train, Z_test, y_train_img, y_test_img,
            class_names=class_names, show=show, save_path=save_transf)


def plt_close(fig: bool | object) -> None:
    if fig is True:
        plt.close("all")
        return
    try:
        plt.close()
    except Exception:
        plt.close("all")
