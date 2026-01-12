from __future__ import annotations

from typing import List, Dict, Tuple, Union, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet50_Weights

from .resnet import ResNet50Backbone
from .gradcam_train import GradCAMPP

from .hyperparams import ExperimentConfig

# Components:
# - Backbone (ResNet50) with classification head
# - Grad-CAM++ module for generating class activation maps and extracting bounding boxes
# - The Grad-CAM++ module hooks into the specified target layer of the backbone
# - The backbone can be frozen to prevent weight updates during training
# - The model can return either classification logits or bounding boxes based on Grad-CAM++ outputs
# - The target layer is specified as a string, allowing flexibility in choosing which layer to use for CAM generation
# - Strict hooks ensure that the target layer exists in the model architecture
class ResNet50GradCamPP(nn.Module):
    def __init__(
        self,
        cfg : ExperimentConfig
    ) -> None:
        super().__init__()
        self.backbone = ResNet50Backbone(
            num_classes=int(cfg.data.num_classes),
            weights=ResNet50_Weights.IMAGENET1K_V2,
            freeze_backbone=False, 
            target="layer4[-1]")
        
        self.campp = GradCAMPP(
            model=self.backbone,
            target_layer=self.backbone.target_layer, 
            strict_hooks=True)

    def forward(
        self,
        x: List[torch.Tensor],
        targets: Optional[List[Dict[str, torch.Tensor]]] = None,
        topk: int = 1, thr: float = 0.35, detach_outputs: bool = True,
    ) -> Tuple[List[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]]:
        x = torch.stack(x, dim=0)
        logits = self.backbone(x)

        # cls = []
        # for t in targets:
        #     y = t["labels"]
        #     cls.append(y.to(torch.long))

        # max_len = max(c.numel() for c in cls)
        # cls_tensor = torch.full((len(cls), max_len), -1, device=x[0].device)
        # for i, c in enumerate(cls):
        #     cls_tensor[i, :c.numel()] = c
        # print(cls_tensor.shape)
        # print(logits.shape)
        loss_dict: Dict[str, torch.Tensor] = {}
        if targets is not None:
            cls = []
            for t in targets:
                y = t.get("labels", None)
                cls.append(int(y[0].item()) if y is not None and y.numel() > 0 else 0)
            cls_tensor = torch.tensor(cls, device=x.device, dtype=torch.long)


            loss_dict["loss"] = F.cross_entropy(logits, cls_tensor, ignore_index=-1)
            class_idx = torch.argmax(logits, dim=1)

            prev = [p.requires_grad for p in self.backbone.parameters()]
            for p in self.backbone.parameters():
                p.requires_grad_(False)
        else:
            class_idx = torch.argmax(logits, dim=1)
            prev = None

        try:
            boxes, labels, scores, valid = self.campp(
                x, class_idx=class_idx, topk=topk, thr=thr,
                use_gradients=True, detach_outputs=detach_outputs)

            outputs: List[Dict[str, torch.Tensor]] = []
            for i in range(x.shape[0]):
                v = valid[i].bool()
                outputs.append({
                    "boxes": boxes[i][v],
                    "labels": labels[i][v],
                    "scores": scores[i][v],
                })
        finally:
            if prev is not None:
                for p, f in zip(self.backbone.parameters(), prev):
                    p.requires_grad_(f)

        return outputs, loss_dict
