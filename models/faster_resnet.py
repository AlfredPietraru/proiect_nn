from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_Weights,
    fasterrcnn_resnet50_fpn,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from .hyperparams import ExperimentConfig


class FasterRCNNResNet50FPN(nn.Module):
    def __init__(
        self,
        num_classes_with_bg: int,
        img_size: int,  # 512 / 640
        weights: Optional[FasterRCNN_ResNet50_FPN_Weights] = None,
        trainable_backbone_layers: int = 3,
        max_det: int = 300,
    ) -> None:
        super().__init__()

        base = fasterrcnn_resnet50_fpn(weights=weights, trainable_backbone_layers=trainable_backbone_layers)
        in_features = base.roi_heads.box_predictor.cls_score.in_features
        base.roi_heads.box_predictor = FastRCNNPredictor(in_features, int(num_classes_with_bg))

        self.backbone = base.backbone
        self.rpn = base.rpn
        self.roi_heads = base.roi_heads
        self.transform = base.transform

        s = int(img_size)
        self.transform.min_size = (s,)
        self.transform.max_size = s
        self.max_det = max_det

    # @property
    # def device(self) -> torch.device:
    #     return next(self.backbone.model.parameters()).device

    def forward(
        self,
        images: Union[torch.Tensor, Sequence[torch.Tensor]],
        targets: Optional[List[Dict[str, torch.Tensor]]] = None,
    ) -> Tuple[List[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]]:
        x_list = self.as_image_list(images)
        # x_list = [im for im in x_list]
        original_sizes = [im.shape[-2:] for im in x_list]

        if targets is not None:
            images_t, targets_t = self.transform(x_list, targets)
        else:
            images_t, targets_t = self.transform(x_list, None)

        feats = self.backbone(images_t.tensors)
        if not isinstance(feats, dict):
            raise TypeError("Backbone must return dict[str, Tensor].")

        proposals, rpn_losses = self.rpn(images_t, feats, targets_t)
        detections, roi_losses = self.roi_heads(feats, proposals, images_t.image_sizes, targets_t)
        outputs = self.transform.postprocess(detections, images_t.image_sizes, original_sizes)

        loss_dict: Dict[str, torch.Tensor] = {}
        if targets is not None:
            loss_dict.update(rpn_losses)
            loss_dict.update(roi_losses)
            loss_dict["total"] = sum(loss_dict.values())

        return outputs, loss_dict

    @staticmethod
    def pack_detections(
        detections: List[Dict[str, torch.Tensor]],
        max_det: int, device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        N = len(detections)
        M = int(max_det)

        boxes_b = torch.zeros((N, M, 4), device=device, dtype=torch.float32)
        labels_b = torch.full((N, M), -1, device=device, dtype=torch.long)
        scores_b = torch.zeros((N, M), device=device, dtype=torch.float32)
        valid_b = torch.zeros((N, M), device=device, dtype=torch.bool)

        for i, det in enumerate(detections):
            if not det:
                continue

            boxes = det.get("boxes", None)
            labels = det.get("labels", None)
            scores = det.get("scores", None)

            if boxes is None or labels is None or scores is None:
                continue
            if boxes.numel() == 0:
                continue

            k = min(M, boxes.shape[0])
            boxes_b[i, :k] = boxes[:k].to(device=device, dtype=torch.float32)
            labels_b[i, :k] = labels[:k].to(device=device, dtype=torch.long)
            scores_b[i, :k] = scores[:k].to(device=device, dtype=torch.float32)
            valid_b[i, :k] = True

        return boxes_b, labels_b, scores_b, valid_b

    def loss(
        self,
        images: Union[torch.Tensor, Sequence[torch.Tensor]],
        targets: List[Dict[str, torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        x_list = self.as_image_list(images)
        x_list = [im.to(self.device, non_blocking=True) for im in x_list]
        targets = [{k: v.to(self.device, non_blocking=True) for k, v in t.items()} for t in targets]

        images_t, targets_t = self.transform(x_list, targets)
        feats = self.backbone(images_t.tensors)
        if not isinstance(feats, dict):
            raise TypeError("Backbone must return dict[str, Tensor].")

        proposals, rpn_losses = self.rpn(images_t, feats, targets_t)
        _, roi_losses = self.roi_heads(feats, proposals, images_t.image_sizes, targets_t)

        out: Dict[str, torch.Tensor] = {}
        out.update(rpn_losses)
        out.update(roi_losses)
        return out

    def as_image_list(
        self, 
        images: Union[torch.Tensor, Sequence[torch.Tensor]]
    ) -> List[torch.Tensor]:
        if torch.is_tensor(images):
            x = images
            if x.ndim != 4:
                raise ValueError("Expected NCHW tensor.")
            return [x[i] for i in range(x.shape[0])]
        if len(images) == 0:
            raise ValueError("Empty image list.")
        return list(images)


def get_model_fasterrcnn(cfg: ExperimentConfig) -> nn.Module:
    model = FasterRCNNResNet50FPN(
        num_classes_with_bg=int(cfg.num_classes_with_bg()),
        img_size=int(cfg.data.img_size),
        weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT,
        trainable_backbone_layers=3,
        max_det=300,
    ).to(torch.device(cfg.train.device))
    s = int(cfg.data.img_size)
    # For FPN - need to set both min and max size
    model.transform.min_size = (s,)
    model.transform.max_size = s
    return model
