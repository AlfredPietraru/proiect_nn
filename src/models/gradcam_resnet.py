from __future__ import annotations

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet50_Weights

from .gradcam_train import GradCAMPP
from .hyperparams import ExperimentConfig
from .resnet import ResNet50Backbone

EPS: float = 1e-6

class ResNet50GradCamPP(nn.Module):
    """
    ResNet50 classifier + GradCAM++ CAM->box.

    Methods:
    - forward: CE loss (if targets) + CAM-derived detections
    - predict_class_logits: (N,C) classifier logits for KDD
    - predict_boxes_logits: packed boxes + logits-like tensor for BoxMatchKDD
    """

    def __init__(
        self,
        num_classes: int, strict_hooks: bool = True,
        weights: ResNet50_Weights = ResNet50_Weights.IMAGENET1K_V2,
        freeze_backbone: bool = False, target: str = "layer4[-1]", max_det: int = 300
    ) -> None:
        super().__init__()

        self.backbone = ResNet50Backbone(num_classes, weights, freeze_backbone, target)
        self.campp = GradCAMPP(self.backbone, self.backbone.target_layer, strict_hooks)
        self.max_det = max_det

    def forward(
        self, x: Tensor, targets: list[dict[str, Tensor]] | dict[str, Tensor] | None = None,
        topk: int = 1, thr: float = 0.35, detach_outputs: bool = True
    ) -> tuple[list[dict[str, Tensor]], dict[str, Tensor]]:
        """Forward pass for training and inference."""
        if isinstance(x, (list, tuple)):
            x = torch.stack(list(x), dim=0)

        logits = self.backbone(x)
        loss_dict: dict[str, Tensor] = {}

        if targets is not None:
            if torch.is_tensor(targets):
                class_targets = targets.to(x.device, torch.long)
            else:
                targets_list: list[dict[str, Tensor]] = targets if isinstance(targets, list) else [targets]
                class_targets = torch.tensor([int(t["labels"][0].item())\
                    if ("labels" in t and t["labels"].numel() > 0) else 0 for t in targets_list
                ], device=x.device, dtype=torch.long)
            loss = F.cross_entropy(logits, class_targets)
            loss_dict = {"loss": loss, "total": loss}

        class_idx = torch.argmax(logits, dim=1)

        with torch.enable_grad():
            x_cam = x if x.requires_grad else x.detach().requires_grad_(True)
            _, boxes, labels, scores, valid = self.campp(x_cam, class_idx, topk, thr, use_gradients=True, detach_outputs=detach_outputs)

        outputs: list[dict[str, Tensor]] = []
        if boxes is None or labels is None or scores is None or valid is None:
            empty_boxes = x.new_zeros((0, 4))
            empty_labels = x.new_zeros((0,), torch.long)
            empty_scores = x.new_zeros((0,), torch.float32)
            for _ in range(x.shape[0]):
                outputs.append({"boxes": empty_boxes, "labels": empty_labels, "scores": empty_scores})
            return outputs, loss_dict

        for i in range(x.shape[0]):
            v = valid[i].bool()
            outputs.append({"boxes": boxes[i][v], "labels": labels[i][v], "scores": scores[i][v]})

        return outputs, loss_dict

    @torch.no_grad()
    def predict_class_logits(self, x: Tensor) -> Tensor:
        """Predict class logits for input tensor."""
        if isinstance(x, (list, tuple)):
            x = torch.stack(list(x), dim=0)
        self.eval()
        return self.backbone(x)

    @torch.no_grad()
    def predict_boxes_logits(self, images: Tensor | list[Tensor], cam_thr: float = 0.35) -> tuple[Tensor, Tensor, Tensor]:
        """Get packed boxes + logits-like tensor for BoxMatchKDD."""
        x = torch.stack(images, 0) if isinstance(images, list) else images
        logits = self.backbone(x)
        class_idx = torch.argmax(logits, dim=1)

        n, c = logits.shape
        m = int(self.max_det)

        with torch.enable_grad():
            x_cam = x.detach().requires_grad_(True)
            _, boxes, labels, scores, valid = self.campp(x_cam, class_idx, m, cam_thr, use_gradients=True, detach_outputs=True)

        boxes_b = x.new_full((n, m, 4), -1.0, torch.float32)
        logits_b = x.new_full((n, m, c), -20.0, torch.float32)
        valid_b = x.new_zeros((n, m), torch.bool)

        if boxes is None or labels is None or scores is None or valid is None:
            return boxes_b, logits_b, valid_b

        k = min(m, boxes.shape[1])
        boxes_b[:, :k] = boxes[:, :k]
        valid_b[:, :k] = valid[:, :k].bool()

        for i, _ in enumerate(images):
            for j, _ in enumerate(range(k)):
                if not bool(valid_b[i, j]):
                    continue
                cls = int(labels[i, j].item())
                if 0 <= cls < c:
                    logits_b[i, j, cls] = torch.log(scores[i, j].clamp_min(EPS))

        return boxes_b, logits_b, valid_b


def get_model_resnet_gradcam(cfg: ExperimentConfig) -> nn.Module:
    return ResNet50GradCamPP(
        num_classes=int(cfg.data.num_classes), strict_hooks=True,
        weights=ResNet50_Weights.IMAGENET1K_V2, freeze_backbone=False,
        target="layer4[-1]", max_det=int(cfg.data.max_objects)
    ).to(torch.device(cfg.train.device))
