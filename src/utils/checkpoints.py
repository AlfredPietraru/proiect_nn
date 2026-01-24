from __future__ import annotations

import os
import torch
import torch.nn as nn
from loguru import logger


def save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer | None, epoch: int, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ckpt = {"epoch": int(epoch), "model_state_dict": model.state_dict()}
    if optimizer is not None:
        ckpt["optimizer_state_dict"] = optimizer.state_dict()

    torch.save(ckpt, path)
    logger.info(f"Checkpoint saved at {path}")


def load_checkpoint(
    path: str, model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    device: torch.device | None = None
) -> tuple[nn.Module, torch.optim.Optimizer | None, int]:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    epoch = int(ckpt.get("epoch", 0))
    logger.info(f"Loaded checkpoint from {path} (epoch={epoch})")
    return model, optimizer, epoch
