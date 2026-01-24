from __future__ import annotations

from typing import cast
import torch
import torch.nn as nn
from torch import Tensor
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights, fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from .hyperparams import ExperimentConfig


def pack_detections(detections: list[dict[str, Tensor]], max_det: int, device: torch.device) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    n, m = len(detections), int(max_det)

    boxes_b  = torch.zeros((n, m, 4), device, torch.float32)
    labels_b = torch.full((n, m), -1, device, torch.long)
    scores_b = torch.zeros((n, m), device, torch.float32)
    valid_b  = torch.zeros((n, m), device, torch.bool)

    for i, det in enumerate(detections):
        boxes, labels, scores = det.get("boxes"), det.get("labels"), det.get("scores")
        if boxes is None or labels is None or scores is None or boxes.numel() == 0:
            continue

        k = min(m, int(boxes.shape[0]))
        boxes_b[i, :k]  = boxes[:k].to(device, torch.float32)
        labels_b[i, :k] = labels[:k].to(device, torch.long)
        scores_b[i, :k] = scores[:k].to(device, torch.float32)
        valid_b[i, :k]  = True

    return boxes_b, labels_b, scores_b, valid_b


class FasterRCNNResNet50FPN(nn.Module):
    """
    TorchVision Faster R-CNN wrapper.

    Methods:
      - forward: training -> outputs + loss_dict; inference -> outputs only
      - loss: compute loss_dict only
      - as_image_list: normalize NCHW tensor or list[Tensor] into list[Tensor]
      - extract_features: backbone features for KDD feature losses
      - predict_boxes_logits: packed detections for BoxMatchKDD
    """

    def __init__(
        self,
        num_classes_with_bg: int, img_size: int,
        weights: FasterRCNN_ResNet50_FPN_Weights | None = None,
        trainable_backbone_layers: int = 3, max_det: int = 300
    ) -> None:
        super().__init__()

        base = fasterrcnn_resnet50_fpn(weights=weights, trainable_backbone_layers=int(trainable_backbone_layers))

        predictor = cast(FastRCNNPredictor, base.roi_heads.box_predictor)
        in_features = predictor.cls_score.in_features
        base.roi_heads.box_predictor = FastRCNNPredictor(in_features, int(num_classes_with_bg))

        self.backbone  = base.backbone
        self.rpn       = base.rpn
        self.roi_heads = base.roi_heads
        self.transform = base.transform

        s = int(img_size)
        self.max_det = int(max_det)
        object.__setattr__(self.transform, "min_size", (s,))
        object.__setattr__(self.transform, "max_size", s)

    def forward(self, images: Tensor | list[Tensor], targets: list[dict[str, Tensor]] | None) -> tuple[list[dict[str, Tensor]], dict[str, Tensor]]:
        x_list = self.as_image_list(images)
        original_sizes = [im.shape[-2:] for im in x_list]

        images_t, targets_t = self.transform(x_list, targets)
        feats = self.backbone(images_t.tensors)
        if not isinstance(feats, dict):
            raise TypeError("Backbone must return dict[str, Tensor].")

        proposals, rpn_losses = self.rpn(images_t, feats, targets_t)
        detections, roi_losses = self.roi_heads(feats, proposals, images_t.image_sizes, targets_t)

        postprocess = self.transform.postprocess
        outputs = postprocess(detections, images_t.image_sizes, original_sizes)

        loss_dict: dict[str, Tensor] = {}
        if targets is not None:
            loss_dict.update(rpn_losses)
            loss_dict.update(roi_losses)
            loss_dict["total"] = sum(loss_dict.values())

        return outputs, loss_dict

    def loss(self, images: Tensor | list[Tensor], targets: list[dict[str, Tensor]]) -> dict[str, Tensor]:
        x_list = self.as_image_list(images)
        images_t, targets_t = self.transform(list(x_list), list(targets))

        feats = self.backbone(images_t.tensors)
        if not isinstance(feats, dict):
            raise TypeError("Backbone must return dict[str, Tensor].")

        proposals, rpn_losses = self.rpn(images_t, feats, targets_t)
        _, roi_losses = self.roi_heads(feats, proposals, images_t.image_sizes, targets_t)

        out: dict[str, Tensor] = {}
        out.update(rpn_losses)
        out.update(roi_losses)
        return out

    def as_image_list(self, images: Tensor | list[Tensor]) -> list[Tensor]:
        if torch.is_tensor(images):
            return [img for _, img in enumerate(images)]
        return list(images)

    def extract_features(self, images: Tensor | list[Tensor]) -> dict[str, Tensor]:
        x_list = self.as_image_list(images)
        images_t, _ = self.transform(list(x_list), None)

        feats = self.backbone(images_t.tensors)
        if not isinstance(feats, dict):
            raise TypeError("Backbone must return dict[str, Tensor].")
        return feats

    def predict_boxes_logits(self, images: Tensor | list[Tensor]) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        x_list = self.as_image_list(images)
        images_t, _ = self.transform(list(x_list), None)

        feats = self.backbone(images_t.tensors)
        if not isinstance(feats, dict):
            raise TypeError("Backbone must return dict[str, Tensor].")

        proposals, _ = self.rpn(images_t, feats, None)
        detections, _ = self.roi_heads(feats, proposals, images_t.image_sizes, None)

        return pack_detections(detections, self.max_det, images_t.tensors.device)


def get_model_fasterrcnn(cfg: ExperimentConfig) -> nn.Module:
    return FasterRCNNResNet50FPN(
        num_classes_with_bg=int(cfg.num_classes_with_bg()),
        img_size=int(cfg.data.img_size), weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT,
        trainable_backbone_layers=3, max_det=int(cfg.data.max_objects)
    ).to(torch.device(cfg.train.device))
