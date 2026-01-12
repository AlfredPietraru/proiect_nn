from __future__ import annotations

from typing import List, Dict, Tuple, Union, Sequence, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet50_Weights

from .resnet import ResNet50Backbone
from .gradcam_train import GradCAMPP
from .hyperparams import ExperimentConfig


class ResNet50GradCamPP(nn.Module):
    def __init__(
        self,
        num_classes: int, strict_hooks: bool = True,
        weights = ResNet50_Weights.IMAGENET1K_V2,
        freeze_backbone: bool = False, target: str = "layer4[-1]"
    ) -> None:
        super().__init__()

        # Components:
        # - Backbone (ResNet50) with classification head
        # - Grad-CAM++ module for generating class activation maps and extracting bounding boxes
        # - The Grad-CAM++ module hooks into the specified target layer of the backbone
        # - The backbone can be frozen to prevent weight updates during training
        # - The model can return either classification logits or bounding boxes based on Grad-CAM++ outputs
        # - The target layer is specified as a string, allowing flexibility in choosing which layer to use for CAM generation
        # - Strict hooks ensure that the target layer exists in the model architecture
        self.backbone = ResNet50Backbone(
            num_classes=num_classes, weights=weights,
            freeze_backbone=freeze_backbone, target=target)
        self.campp = GradCAMPP(
            model=self.backbone,
            target_layer=self.backbone.target_layer, 
            strict_hooks=strict_hooks)

    def forward(
        self,
        x: torch.Tensor,
        targets: Optional[Union[torch.Tensor, List[Dict[str, torch.Tensor]]]] = None,
        topk: int = 1, thr: float = 0.35, detach_outputs: bool = True,
    ) -> Tuple[List[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]]:
        if x.ndim != 4:
            raise ValueError("Input must be NCHW")

        logits = self.backbone(x)

        loss_dict: Dict[str, torch.Tensor] = {}
        if targets is not None:
            if torch.is_tensor(targets):
                class_targets = targets
            else:
                cls = []
                for t in targets:
                    y = t.get("labels", None)
                    cls.append(int(y[0].item()) if y is not None and y.numel() > 0 else 0)
                class_targets = torch.tensor(cls, device=x.device, dtype=torch.long)

            loss_dict = {}
            loss_dict["loss"] = F.cross_entropy(logits, class_targets)

        class_idx = torch.argmax(logits, dim=1)
        prev = [p.requires_grad for p in self.backbone.parameters()]
        for p in self.backbone.parameters():
            p.requires_grad_(False)

        try:
            boxes, labels, scores, valid = self.campp(
                x, class_idx=class_idx, topk=topk, thr=thr,
                use_gradients=True, detach_outputs=detach_outputs)

            outputs: List[Dict[str, torch.Tensor]] = []
            for i in range(x.shape[0]):
                v = valid[i].bool()
                outputs.append({"boxes": boxes[i][v], "labels": labels[i][v], "scores": scores[i][v]})
        finally:
            for p, f in zip(self.backbone.parameters(), prev):
                p.requires_grad_(f)

        return outputs, loss_dict

    def extract_features(
        self,
        images: Union[torch.Tensor, Sequence[torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        x_list = self.as_image_list(images)
        x_list = [im for im in x_list]

        images_t, _ = self.transform(x_list, None)
        feats = self.backbone(images_t.tensors)
        if not isinstance(feats, dict):
            raise TypeError("Backbone must return dict[str, Tensor].")

        return feats

    def predict_boxes_logits(
        self,
        images: Union[torch.Tensor, Sequence[torch.Tensor]],
        cam_thr: float = 0.35
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x_list = self.as_image_list(images)
        x_list = [im for im in x_list]

        images_t, _ = self.transform(x_list, None)
        logits = self.backbone(images_t.tensors)

        N, M = images_t.tensors.shape[0], self.campp.max_det
        boxes_b = images_t.tensors.new_zeros((N, M, 4), dtype=torch.float32)
        labels_b = images_t.tensors.new_full((N, M), -1, dtype=torch.long)
        scores_b = images_t.tensors.new_zeros((N, M), dtype=torch.float32)
        valid_b = images_t.tensors.new_zeros((N, M), dtype=torch.bool)

        class_idx = torch.argmax(logits, dim=1)

        pred_boxes, pred_cls, pred_scores, pred_valid = self.campp(
            images_t.tensors,
            class_idx=class_idx,
            topk=M, threshold=cam_thr,
            use_gradients=False, detach_outputs=True)

        k = min(M, pred_boxes.shape[1])
        if k > 0:
            boxes_b[:, :k, :] = pred_boxes[:, :k, :]
            labels_b[:, :k] = pred_cls[:, :k]
            scores_b[:, :k] = pred_scores[:, :k]
            valid_b[:, :k] = pred_valid[:, :k]

        return boxes_b, labels_b, scores_b, valid_b


def get_model_resnet_gradcam(cfg: ExperimentConfig) -> nn.Module:
    return ResNet50GradCamPP(
        num_classes=int(cfg.data.num_classes),
        strict_hooks=True, weights=ResNet50_Weights.IMAGENET1K_V2,
        freeze_backbone=False, target="layer4[-1]"
    ).to(torch.device(cfg.train.device))
