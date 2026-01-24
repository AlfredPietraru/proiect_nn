from __future__ import annotations

from torch import Tensor
import torch.nn as nn
from torchvision.models import ResNet50_Weights, resnet50


class ResNet50Backbone(nn.Module):
    def __init__(
        self,
        num_classes: int,
        weights: ResNet50_Weights | None = ResNet50_Weights.IMAGENET1K_V2,
        # Layer 4 - best for obtaining larger receptive field CAMs for object detection
        # Layer 3 - better for smaller objects (discussed in Grad-CAM++ paper)
        freeze_backbone: bool = False, target: str = "layer4[-1]"  # layer3[-1], layer4[-1].conv3, layer4[-1].relu
    ) -> None:
        super().__init__()

        base = resnet50(weights=weights)
        in_feats = base.fc.in_features
        base.fc = nn.Linear(in_feats, num_classes, bias=True)
        self.model = base

        if freeze_backbone:
            for name, p in self.model.named_parameters():
                if not name.startswith("fc."):
                    p.requires_grad_(False)

        self.target_layer = self.resolve_target_layer(target)

    def resolve_target_layer(self, spec: str) -> nn.Module:
        if spec == "layer4[-1]":
            return self.model.layer4[-1]
        if spec == "layer4[-1].conv3":
            return self.model.layer4[-1].conv3
        if spec == "layer4[-1].relu":
            return self.model.layer4[-1].relu

        raise ValueError("Bad target. Use: layer4[-1], layer4[-1].conv3, layer4[-1].relu")

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 4:
            raise ValueError("Input must be NCHW")
        return self.model(x)
