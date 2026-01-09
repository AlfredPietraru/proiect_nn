from __future__ import annotations

from typing import Dict, List
import torch


def move_images_to_device(images: List[torch.Tensor], device: torch.device) -> List[torch.Tensor]:
    return [img.to(device, non_blocking=True) for img in images]


def move_targets_to_device(targets: List[dict], device: torch.device) -> List[dict]:
    for t in targets:
        t["boxes"] = t["boxes"].to(device, non_blocking=True)
        t["labels"] = t["labels"].to(device, non_blocking=True)
        if "scores" in t and t["scores"] is not None:
            t["scores"] = torch.tensor(t["scores"], dtype=torch.float32).to(device, non_blocking=True)
    return targets


def mean_history(history: Dict[str, float], steps: int) -> Dict[str, float]:
    denom = max(1, steps)
    return {k: v / denom for k, v in history.items()}
