from __future__ import annotations

from models.faster_resnet import get_model_fasterrcnn
from models.gradcam_resnet import get_model_resnet_gradcam
from models.yolon11 import get_model_yolo11
from models.hyperparams import ExperimentConfig

import torch


def build_model(cfg: ExperimentConfig) -> torch.nn.Module:
    arch = getattr(cfg.model, "arch")

    if arch == "fasterrcnn":
        return get_model_fasterrcnn(cfg=cfg)
    if arch == "resnet50_gradcampp":
        return get_model_resnet_gradcam(cfg=cfg)
    if arch == "yolo11":
        return get_model_yolo11(cfg=cfg)

    raise ValueError(f"Unknown Model Architecture: {arch}")
