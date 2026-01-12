from __future__ import annotations

from typing import List, Dict, Tuple, Optional, Union

import torch
import torch.nn as nn
from ultralytics import YOLO

from .hyperparams import ExperimentConfig


class YOLO11Detector(nn.Module):
    def __init__(
        self, num_classes: int, imgsz: int,  # 512 / 640
        conf: float, iou: float,  weights_path: str = "yolo11n.pt",
        max_det: int = 300, agnostic_nms: bool = True, device: Optional[str] = None
    ) -> None:
        super().__init__()

        # Components:
        # - YOLOv11 model loaded from specified weights path
        # - Configurable image size, confidence threshold, IoU threshold for predictions
        # - Maximum number of detections per image
        # - Option for class-agnostic NMS during inference (useful for pseudo-labeling)
        # - The model is designed to output bounding boxes, class labels, scores, and validity masks for detections
        # - The model can be moved to different devices (CPU/GPU) using the `to` method
        # - Inference is performed in a no-grad context to save memory and computation
        self.yolo = YOLO(weights_path)

        self.num_classes = num_classes
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.max_det = max_det
        self.agnostic_nms = agnostic_nms
        self.device = torch.device(device) if device is not None else torch.device("cpu")

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.yolo.to(*args, **kwargs)
        return self

    @torch.inference_mode()
    def forward(
        self,
        x: torch.Tensor,
        targets: Optional[List[Dict[str, torch.Tensor]]] = None
    ) -> Tuple[List[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]]:
        if targets is not None:
            raise NotImplementedError("YOLO11Detector does not support training mode.")

        boxes_b, labels_b, scores_b, valid_b = self.predict_packed(x)

        outputs: List[Dict[str, torch.Tensor]] = []
        for i in range(x.shape[0]):
            v = valid_b[i]
            outputs.append({
                "boxes": boxes_b[i][v],
                "labels": labels_b[i][v],
                "scores": scores_b[i][v],
            })

        return outputs, {}
    
    @torch.inference_mode()
    def predict_packed(
        self, 
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if x.ndim != 4:
            raise ValueError("Need NCHW")

        if x.device != self.device:
            x = x.to(self.device, non_blocking=True)

        results = self.yolo.predict(
            source=x, imgsz=self.imgsz, conf=self.conf, iou=self.iou,
            max_det=self.max_det, agnostic_nms=self.agnostic_nms,
            device=str(self.device), verbose=False)

        N, M = x.shape[0], self.max_det
        boxes_b = x.new_zeros((N, M, 4), dtype=torch.float32)
        labels_b = x.new_full((N, M), -1, dtype=torch.long)
        scores_b = x.new_zeros((N, M), dtype=torch.float32)
        valid_b = x.new_zeros((N, M), dtype=torch.bool)

        for i, r in enumerate(results):
            b = r.boxes
            if b is None:
                continue

            xyxy = getattr(b, "xyxy", None)
            if xyxy is None:
                continue

            boxes = torch.tensor(xyxy, device=self.device, dtype=torch.float32)
            if boxes.numel() == 0:
                continue

            labels = torch.tensor(getattr(b, "cls", []), device=self.device, dtype=torch.long)
            scores = torch.tensor(getattr(b, "conf", []), device=self.device, dtype=torch.float32)

            k = min(M, boxes.shape[0])
            if k == 0:
                continue

            boxes_b[i, :k] = boxes[:k]
            labels_b[i, :k] = labels[:k]
            scores_b[i, :k] = scores[:k]
            valid_b[i, :k] = True

        return boxes_b, labels_b, scores_b, valid_b

    def extract_features(
        self,
        images: Union[torch.Tensor, List[torch.Tensor]]
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
        images: Union[torch.Tensor, List[torch.Tensor]]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x_list = self.as_image_list(images)
        x_list = [im for im in x_list]

        images_t, _ = self.transform(x_list, None)
        logits = self.backbone(images_t.tensors)

        N, M = images_t.tensors.shape[0], self.max_det
        boxes_b = images_t.tensors.new_zeros((N, M, 4), dtype=torch.float32)
        labels_b = images_t.tensors.new_full((N, M), -1, dtype=torch.long)
        scores_b = images_t.tensors.new_zeros((N, M), dtype=torch.float32)
        valid_b = images_t.tensors.new_zeros((N, M), dtype=torch.bool)

        class_idx = torch.argmax(logits, dim=1)
        labels_b.fill_(class_idx.view(-1, 1))        
        boxes_b, labels_b, scores_b, valid_b = self.predict_packed(images_t.tensors)

        return boxes_b, labels_b, scores_b, valid_b


def get_model_yolo11(cfg: ExperimentConfig) -> nn.Module:
    return YOLO11Detector(
        num_classes=int(cfg.data.num_classes),
        imgsz=int(cfg.data.img_size),
        weights_path="yolo11n.pt",
        conf=float(cfg.ssl.pseudo_conf_thr),
        iou=float(cfg.ssl.match_iou_thr),
        max_det=300,  # => in an image can be at most 300 objects
        agnostic_nms=True,  # => class-agnostic NMS (for pseudo-labels)
        # keep_labels=True/False => not exposed (it acts as a mask during training)
    ).to(torch.device(cfg.train.device))
