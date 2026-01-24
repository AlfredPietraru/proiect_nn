from __future__ import annotations

from typing import Any
import torch
from torch.utils.data import Dataset, DataLoader

from utils import pick_workers


def move_images_to_device(images: list[torch.Tensor], device: torch.device) -> list[torch.Tensor]:
    """Move image tensors to device."""
    return [img.to(device, non_blocking=True) for img in images]


def move_targets_to_device(targets: list[dict], device: torch.device) -> list[dict]:
    """Move target dicts to device."""
    for t in targets:
        t["boxes"] = t["boxes"].to(device, non_blocking=True)
        t["labels"] = t["labels"].to(device, non_blocking=True)
        if "scores" in t and t["scores"] is not None:
            t["scores"] = torch.tensor(t["scores"], dtype=torch.float32).to(device, non_blocking=True)
    return targets


def mean_history(history: dict[str, float], steps: int) -> dict[str, float]:
    """Mean the history values over the number of steps."""
    denom = max(1, steps)
    return {k: v / denom for k, v in history.items()}


def collate_clone_images(batch: list[Any]) -> tuple[torch.Tensor, list[Any]]:
    """
    Collate (img, target) items.
    - clones images to ensure resizable storage for stacking
    - leaves targets as a list (detection-style)
    """
    images, targets = zip(*batch)

    imgs_out: list[torch.Tensor] = []
    for im in images:
        if not torch.is_tensor(im):
            im = torch.as_tensor(im)
        # Make storage resizable + contiguous
        im = im.contiguous()
        if not im.is_contiguous():
            im = im.contiguous()
        # The key line: force owning storage
        im = im.clone()
        imgs_out.append(im)

    return torch.stack(imgs_out, dim=0), list(targets)


def stats_mean_std(dataset: Dataset, max_samples: int = 3000) -> tuple[list[float], list[float]]:
    """Mean and standard deviation calculation for image datasets."""
    loader = DataLoader(
        dataset, batch_size=64, shuffle=False,
        num_workers=pick_workers(),
        collate_fn=collate_clone_images,
        pin_memory=False, persistent_workers=False)

    mean = torch.zeros(3, dtype=torch.float64)
    mean_sq = torch.zeros(3, dtype=torch.float64)
    total_pixels = 0
    processed_samples = 0

    for imgs, _ in loader:
        imgs = imgs.float()

        if imgs.max() > 1.0:
            imgs = imgs / 255.0

        b, c, h, w = imgs.shape
        if c != 3:
            raise ValueError(f"Expected 3 channels, got {c}")

        num_pixels = b * h * w
        sum_per_channel = imgs.sum(dim=(0, 2, 3), dtype=torch.float64)
        sum_sq_per_channel = (imgs * imgs).sum(dim=(0, 2, 3), dtype=torch.float64)

        mean += sum_per_channel
        mean_sq += sum_sq_per_channel
        total_pixels += num_pixels
        processed_samples += b

        if processed_samples >= max_samples:
            break

    overall_mean = mean / float(total_pixels)
    overall_var = (mean_sq / float(total_pixels)) - (overall_mean ** 2)
    overall_var = torch.clamp(overall_var, min=0.0)
    overall_std = torch.sqrt(overall_var)

    return overall_mean.tolist(), overall_std.tolist()
