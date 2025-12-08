import copy
import math
import warnings
from typing import Iterable, Optional

import torch
import torch.nn as nn


class RobustEMA:
    """
    Robust Exponential Moving Average (EMA) helper for model weights (teacher).
    - Creates a deep copy of `model` as the EMA (teacher) model.
    - Disables gradients for EMA model params.
    - Updates EMA after an optimizer.step() call using a warm-up decay schedule.
    - Hard-copies BatchNorm running stats & non-float buffers; applies EMA to floats.
    - Safely handles FP16 by casting model values to ema dtype before update.
    - Can copy arbitrary non-parameter attributes from online model via update_attr.
    """

    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.9999,
        tau: Optional[float] = 2000.0,
        device: Optional[torch.device] = None,
        copy_buffers_hard: bool = True,
        warn_if_mismatch: bool = True,
    ):
        assert 0.0 <= decay < 1.0, "decay must be in [0,1)"
        if tau is not None:
            assert tau > 0.0, "tau must be positive or None"

        self.decay = float(decay)
        self.tau = tau
        self.updates = 0  # number of EMA updates performed
        self.warn_if_mismatch = warn_if_mismatch
        self.copy_buffers_hard = copy_buffers_hard

        # Deep copy model -> ema model (teacher)
        self.ema = copy.deepcopy(model).eval()
        if device is not None:
            self.ema.to(device)

        # Disable grads for EMA model
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def _get_decay(self) -> float:
        """
        Compute decay for current update step with warm-up.
        We use a standard warm-up: d = decay * (1 - exp(-t / tau)), clipped to [0, decay].
        If tau is None, return decay.
        """
        if self.tau is None:
            return self.decay
        # warm-up factor in (0,1)
        warmup = 1.0 - math.exp(-self.updates / float(self.tau))
        d = self.decay * warmup
        # numeric safety
        d = min(max(d, 0.0), self.decay)
        return d

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """
        Update EMA params from `model` (the online model).
        Must be called AFTER optimizer.step().
        """
        self.updates += 1
        d = self._get_decay()  # decay factor in [0, decay]

        ema_state = self.ema.state_dict()
        model_state = model.state_dict()

        # Iterate keys from model_state (safe if keys missing in ema, warn)
        for name, model_val in model_state.items():
            if name not in ema_state:
                if self.warn_if_mismatch:
                    warnings.warn(f"EMA: key '{name}' present in model but missing in ema model; skipping.")
                continue

            ema_val = ema_state[name]

            # Move model_val to ema device/dtype for safe ops
            try:
                model_val_device = model_val.to(ema_val.device)
            except Exception:
                model_val_device = model_val

            # Hard-copy non-float tensors (e.g., integers, counters)
            if not ema_val.dtype.is_floating_point:
                ema_val.copy_(model_val_device)
                continue

            # Common BN-running buffers should be hard-copied rather than EMA'd
            # (running_mean, running_var, num_batches_tracked)
            if self.copy_buffers_hard and (
                name.endswith("running_mean")
                or name.endswith("running_var")
                or name.endswith("num_batches_tracked")
            ):
                ema_val.copy_(model_val_device)
                continue

            # Finally, apply EMA update for floating tensors.
            # Cast model val to ema dtype (handles fp16 -> fp32 safety)
            model_val_cast = model_val_device.to(dtype=ema_val.dtype)

            # ema = d * ema + (1-d) * model
            # alpha = (1 - d)
            alpha = 1.0 - d
            ema_val.mul_(d).add_(model_val_cast, alpha=alpha)

    @torch.no_grad()
    def update_attr(
        self,
        model: nn.Module,
        include: Iterable[str] = (),
        exclude: Iterable[str] = ("process_group", "reducer"),
    ) -> None:
        """
        Copy selected attributes from the online model to the EMA model.
        Useful for non-parameter model state (anchors, strides, meta, cfg).
        - include: if non-empty, only keys in 'include' are copied.
        - exclude: ignore attributes containing any substring in exclude.
        """
        for k, v in model.__dict__.items():
            if include and k not in include:
                continue
            if any(x in k for x in exclude):
                continue
            try:
                setattr(self.ema, k, copy.deepcopy(v))
            except Exception:
                # Best-effort: try shallow set and warn
                try:
                    setattr(self.ema, k, v)
                except Exception as e:
                    warnings.warn(f"EMA.update_attr: could not copy attribute '{k}': {e}")

    def state_dict(self) -> dict:
        """Return a dict that can be saved. Contains ema state_dict and bookkeeping."""
        return {
            "ema_state_dict": self.ema.state_dict(),
            "updates": self.updates,
            "decay": self.decay,
            "tau": self.tau,
        }

    def load_state_dict(self, sd: dict, map_location: Optional[torch.device] = None):
        """
        Load EMA state. Accepts `sd` with keys from state_dict().
        map_location: optional device to map EMA model to.
        """
        if "ema_state_dict" in sd:
            state = sd["ema_state_dict"]
            # map tensors to ema device if specified
            if map_location is not None:
                # load into ema model (which may already be on some device)
                self.ema.load_state_dict(torch.load(state, map_location=map_location) if isinstance(state, str) else state)
            else:
                self.ema.load_state_dict(state)
        else:
            raise KeyError("state_dict must contain 'ema_state_dict'")

        self.updates = int(sd.get("updates", self.updates))
        self.decay = float(sd.get("decay", self.decay))
        self.tau = sd.get("tau", self.tau)

    def to(self, device: torch.device):
        """Move EMA model to device."""
        self.ema.to(device)

    def __repr__(self):
        return f"RobustEMA(decay={self.decay}, tau={self.tau}, updates={self.updates})"


# ---------------------
# Example usage snippet
# ---------------------
#
# model = get_model(...)
# ema = RobustEMA(model, decay=0.9999, tau=2000.0, device=device)
# optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
#
# for data in loader:
#     loss = ...
#     optimizer.zero_grad(); loss.backward(); optimizer.step()
#     # Update EMA after optimizer.step()
#     ema.update(model)
#     # if some metadata changed (anchors, strides), sync attributes:
#     ema.update_attr(model, include=("anchor_generator", "strides", "num_classes"))
#
# # Save
# torch.save({"model_state": model.state_dict(),
#             "ema_state": ema.state_dict(),
#             "optimizer": optimizer.state_dict()}, "chkpt.pth")
#
# # Load (example)
# chk = torch.load("chkpt.pth", map_location=device)
# model.load_state_dict(chk["model_state"])
# ema.load_state_dict(chk["ema_state"], map_location=device)
