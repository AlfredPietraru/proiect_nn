from __future__ import annotations

import copy
from typing import Any

import torch
import torch.nn as nn


def unwrap_model(m: nn.Module) -> nn.Module:
    module = getattr(m, "module", None)
    if isinstance(module, nn.Module):
        return module
    return m


class EMA:
    def __init__(
        self,
        teacher_model: nn.Module, decay: float = 0.9996, 
        init_from: nn.Module | None = None, adaptive: bool = False, 
        alpha_min: float = 0.99, alpha_max: float = 0.9999
    ) -> None:
        if not (0.0 <= decay < 1.0):
            raise ValueError("Decay must be in [0,1)")
        if not (0.0 < alpha_min < 1.0 and 0.0 < alpha_max < 1.0 and alpha_min <= alpha_max):
            raise ValueError("Alpha min/max must be in (0,1) and min <= max")

        base = unwrap_model(teacher_model)
        self.ema = copy.deepcopy(base).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)

        if init_from is not None:
            self.copy_matching(self.ema, unwrap_model(init_from))

        self.decay = float(decay)
        self.adaptive = bool(adaptive)
        self.alpha_min = float(alpha_min)
        self.alpha_max = float(alpha_max)

    def model(self) -> nn.Module:
        return self.ema

    def resolve_alpha(self, conf_mean: float | None) -> float:
        if not self.adaptive or conf_mean is None:
            return self.decay
        c = float(max(0.0, min(1.0, conf_mean)))
        return self.alpha_min + (self.alpha_max - self.alpha_min) * c

    @torch.no_grad()
    def update(self, student_model: nn.Module, conf_mean: float | None) -> tuple[int, int, int]:
        student = unwrap_model(student_model)

        ema_sd = self.ema.state_dict()
        stu_sd = student.state_dict()

        alpha = self.resolve_alpha(conf_mean)
        one_minus = 1.0 - alpha

        updated, missing, shape_mismatch = 0, 0, 0

        for k, ema_v in ema_sd.items():
            stu_v = stu_sd.get(k, None)
            if stu_v is None:
                missing += 1
                continue
            if ema_v.shape != stu_v.shape:
                shape_mismatch += 1
                continue

            if ema_v.dtype.is_floating_point and stu_v.dtype.is_floating_point:
                ema_v.mul_(alpha).add_(
                    stu_v.detach().to(ema_v.device, ema_v.dtype),
                    alpha=one_minus)
            else:
                ema_v.copy_(
                    stu_v.detach().to(ema_v.device, ema_v.dtype))

            updated += 1

        self.ema.load_state_dict(ema_sd, strict=False)
        return updated, missing, shape_mismatch

    def state_dict(self) -> dict[str, Any]:
        return {
            "decay": self.decay, "adaptive": self.adaptive,
            "alpha_min": self.alpha_min, "alpha_max": self.alpha_max,
            "ema": self.ema.state_dict()}

    def load_state_dict(self, sd: dict[str, Any]) -> None:
        self.decay = float(sd.get("decay", self.decay))
        self.adaptive = bool(sd.get("adaptive", self.adaptive))
        self.alpha_min = float(sd.get("alpha_min", self.alpha_min))
        self.alpha_max = float(sd.get("alpha_max", self.alpha_max))
        self.ema.load_state_dict(sd["ema"], strict=False)

    @staticmethod
    @torch.no_grad()
    def copy_matching(dst: nn.Module, src: nn.Module) -> None:
        dst_sd = dst.state_dict()
        src_sd = src.state_dict()
        for k, v in dst_sd.items():
            sv = src_sd.get(k, None)
            if sv is None or sv.shape != v.shape:
                continue
            v.copy_(sv.detach().to(v.device, v.dtype))
        dst.load_state_dict(dst_sd, strict=False)
