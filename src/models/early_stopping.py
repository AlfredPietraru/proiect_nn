from __future__ import annotations

from typing import Any, cast
import copy
from loguru import logger

import math
import torch
import torch.nn as nn


class EarlyStopping:
    def __init__(self, patience: int = 4, min_delta: float = 1e-3, mode: str = "min", verbose: bool = True) -> None:
        assert mode in ("min", "max"), "Mode must be 'min' or 'max'"

        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose

        self.best_value: float | None = None
        self.best_epoch: int | None = None
        self.counter: int = 0
        self.early_stop: bool = False
        self.best_model_state: dict[str, Any] | None = None

    def is_improvement(self, value: float) -> bool:
        if self.best_value is None:
            return True

        if self.mode == "min":
            return value < (self.best_value - self.min_delta)

        return value > (self.best_value + self.min_delta)

    def __call__(self, value: float, model: Any, epoch: int | None) -> None:
        if value is None or (isinstance(value, float) and (math.isnan(value) or math.isinf(value))):
            if self.verbose:
                logger.info("Early stopping received NaN/Inf metric, stopping training.")
            self.early_stop = True
            return

        value_f = float(value)

        if self.is_improvement(value_f):
            self.best_value = value_f
            self.best_epoch = epoch
            self.counter = 0

            if hasattr(model, "state_dict") and callable(model.state_dict):
                state = cast(dict[str, Any], model.state_dict())
                self.best_model_state = {
                    k: (v.detach().cpu().clone() if torch.is_tensor(v) else copy.deepcopy(v))
                    for k, v in state.items()}

            if self.verbose:
                msg = f"Early stopping detected new best {self.mode} value = {self.best_value:.6f}"
                if epoch is not None:
                    msg += f" at epoch {epoch}"
                logger.info(msg)
            return

        self.counter += 1
        if self.verbose:
            best_str = f"{self.best_value:.6f}" if self.best_value is not None else "None"
            logger.info(f"Early stopping counter: {self.counter}/{self.patience} (best={best_str})")

        if self.counter >= self.patience:
            self.early_stop = True
            if self.verbose:
                logger.info("Early stopping stopped training")

    def load_best_model(self, model: nn.Module) -> None:
        if self.best_model_state is None:
            return
        try:
            model.load_state_dict(self.best_model_state, strict=True)
        except RuntimeError as e:
            logger.warning(f"Warning: incompatible with current model. Skipping restore.\n{e}")
