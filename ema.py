import copy
import math
from typing import Iterable

import torch
import torch.nn as nn


class EMA:
    """
    Exponential Moving Average (EMA) of model parameters.
    NOTE:
    - The EMA model is a *shadow model* used only for evaluation/inference.
    - It should NOT receive gradients.
    - Its parameters are updated ONLY from the online model via EMA formula:
            ema = decay * ema + (1 - decay) * online
    """

    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.9999,
        tau: float | None = 2000.0,
        device: torch.device | None = None,
    ) -> None:
        assert 0.0 <= decay < 1.0, "EMA decay must be in [0.0, 1.0)"
        assert tau is None or tau > 0.0, "EMA tau must be positive or None"

        self.decay = float(decay)
        self.tau = tau

        # Counts how many EMA updates have happened so far
        # Used for warm-up of the EMA decay factor
        self.updates: int = 0

        # Create deep copy of the model.
        # EMA model (teacher / shadow model).
        self.ema: nn.Module = copy.deepcopy(model).eval()
        if device is not None:
            self.ema.to(device)

        #! CRITICAL:
        #! Disable gradient updates for the EMA model.
        #! Why?
        # - The EMA model must NOT be optimized by gradient descent.
        # - It must only get updated through EMA smoothing from online model.
        # - If gradients touched EMA, it would stop being a true moving average.
        for p in self.ema.parameters():
            p.requires_grad_(False)

    # Warm-up EMA decay.
    # For the first iterations, EMA uses a smaller decay to catch up
    # faster to the online model. After many updates it approaches 'decay'.
    def update_decay(self) -> float:
        if self.tau is None or self.tau <= 0.0:
            return self.decay
        # Warm-up formula:
        #   D(t) = decay * (1 - exp(-t / tau))
        return self.decay * (1.0 - math.exp(-self.updates / self.tau))

    @torch.no_grad()  # EMA must not require gradient tracking
    def update(self, model: nn.Module) -> None:
        """
        Update EMA parameters from the online model parameters.
        IMPORTANT:
        - CALLED ONLY AFTER optimizer.step() propagates updates to online model.
        - Performs: ema = d * ema + (1-d) * model
        - Does NOT modify the online model.
        """

        # Increase update count
        self.updates += 1

        # Decay factor for this update
        d = self.update_decay()
        ema_state = self.ema.state_dict()     # shadow model state
        model_state = model.state_dict()      # online model state

        # Update each parameter/buffer
        for key, ema_val in ema_state.items():
            model_val = model_state[key]
            # Non-floating tensors (e.g., integers, counters)
            # are simply copied directly.
            # EMA should only be applied to floating weights.
            if not ema_val.dtype.is_floating_point:
                ema_val.copy_(model_val)
                continue

            # EMA update:
            #   ema = d * ema + (1 - d) * model
            alpha = 1.0 - d
            ema_val.mul_(d).add_(model_val.detach(), alpha=alpha)

    @torch.no_grad()
    def update_attr(
        self,
        model: nn.Module,
        include: Iterable[str] = (),
        exclude: Iterable[str] = ("process_group", "reducer"),
    ) -> None:
        """
        Copy selected attributes from the online model to the EMA model.
        Useful for:
        - step counters
        - class names / label metadata
        - anchors / strides
        - configuration fields
        Does NOT touch the trainable parameters.
        """
        for k, v in model.__dict__.items():
            # If include is non-empty, only copy keys inside it
            if include and k not in include:
                continue
            # Avoid copying internal DDP/distributed training fields
            if any(x in k for x in exclude):
                continue
            setattr(self.ema, k, v)


def build_ema(
    model: nn.Module,
    decay: float = 0.9999,
    tau: float | None = 2000.0,
    device: torch.device | None = None,
) -> EMA:
    return EMA(model=model, decay=decay, tau=tau, device=device)
