from __future__ import annotations

import os
import torch
import warnings

from models.hyperparams import ExperimentConfig
from burn_in import pipeline_burn_in
from unbiased_teacher import pipeline_semi_supervised
from utils.oncuda import set_seed
from data.dataloaders import build_dataloaders

warnings.filterwarnings("ignore", category=UserWarning)


if __name__ == "__main__":
    cfg = ExperimentConfig()
    set_seed(cfg.seed)

    device = torch.device(cfg.train.device)

    checkpoint_dir = "./models/checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    data = build_dataloaders(cfg)

    # FASTER-RCNN (rpn) LOSS TERMS
    METRIC_BURN_IN = ["loss_classifier", "loss_box_reg", "loss_objectness", "loss_rpn_box_reg", "total"]
    METRIC_SUP = ["loss_classifier", "loss_box_reg", "loss_objectness", "loss_rpn_box_reg"]
    METRIC_UNSUP = ["loss_classifier", "loss_objectness"]
    VALIDATION_METRICS = ["mAP_50", "mAP_5095", "precision", "recall", "f1"]

    pipeline_burn_in(cfg=cfg, data=data, device=device, metric_keys=METRIC_BURN_IN)

    last_ckpt = os.path.join(checkpoint_dir, f"checkpoint_epoch_{cfg.train.epochs}.pth")
    pipeline_semi_supervised(
        cfg=cfg, checkpoint_path=last_ckpt, data=data, device=device,
        metric_sup=METRIC_SUP, metric_unsup=METRIC_UNSUP, eval_metrics=VALIDATION_METRICS)
