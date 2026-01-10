from __future__ import annotations

import math
from typing import Iterable, List

import torch


def clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


def warmup_scale(it: int, Tw: int, warmup_bias_lr: float) -> float:
    if Tw <= 0 or it >= Tw:
        return 1.0
    if not (0.0 <= warmup_bias_lr <= 1.0):
        raise ValueError("Warmup bias lr must be in [0, 1].")
    t = it / float(Tw)
    return warmup_bias_lr + (1.0 - warmup_bias_lr) * t


def cosine_lr_lambda(
    it: int, T: int, Tw: int, 
    warmup_bias_lr: float, min_lr_ratio: float) -> float:
    if not (0.0 <= min_lr_ratio <= 1.0):
        raise ValueError("Min lr ratio must be in [0, 1].")
    if it < Tw:
        return warmup_scale(it, Tw, warmup_bias_lr)
    t = clamp01((it - Tw) / float(max(1, T - Tw)))
    return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * t))


def onecycle_lr_lambda(
    it: int, T: int, Tw: int,
    warmup_bias_lr: float, min_lr_ratio: float) -> float:
    return cosine_lr_lambda(it, T, Tw, warmup_bias_lr, min_lr_ratio)


def multistep_lr_lambda(
    it: int, T: int, Tw: int, 
    warmup_bias_lr: float, milestones_iters: Iterable[int], gamma: float) -> float:
    if gamma <= 0.0:
        raise ValueError("Gamma must be > 0.")
    if it < Tw:
        return warmup_scale(it, Tw, warmup_bias_lr)
    drops = 0
    for m in milestones_iters:
        if it >= int(m):
            drops += 1
    return gamma**drops


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    scheme: str, total_epochs: int,
    steps_per_epoch: int, milestones: List[int],
    warmup_epochs: int = 3, warmup_bias_lr: float = 0.1,
    min_lr_ratio: float = 0.005, gamma: float = 0.1
) -> torch.optim.lr_scheduler._LRScheduler:
    scheme = scheme.lower().strip()

    if total_epochs <= 0 or steps_per_epoch <= 0:
        raise ValueError("No. epochs and steps per epoch must be > 0.")
    if warmup_epochs < 0:
        raise ValueError("Warmup epochs must be >= 0.")

    T = max(1, int(total_epochs * steps_per_epoch))
    Tw = max(0, int(warmup_epochs * steps_per_epoch))

    if scheme == "multistep":
        milestones_iters = [int(m * steps_per_epoch) for m in milestones]
        def fn(it: int) -> float:
            return multistep_lr_lambda(it, T, Tw, warmup_bias_lr, milestones_iters, gamma)
    elif scheme in ("cosine", "lambda"):
        def fn(it: int) -> float:
            return cosine_lr_lambda(it, T, Tw, warmup_bias_lr, min_lr_ratio)
    elif scheme == "onecycle":
        def fn(it: int) -> float:
            return onecycle_lr_lambda(it, T, Tw, warmup_bias_lr, min_lr_ratio)
    else:
        raise ValueError(f"Unsupported scheduler scheme='{scheme}'")

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=fn)
