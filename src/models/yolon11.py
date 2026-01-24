from __future__ import annotations

import torch
from torch import Tensor
import torch.nn as nn
from ultralytics import YOLO

from .hyperparams import ExperimentConfig

EPS: float = 1e-6

class YOLO11Detector(nn.Module):
    """
    YOLOv11 wrapper.

    Methods:
    - to: move wrapper + ultralytics model to device
    - forward: training -> returns loss_dict; inference -> returns detections list
    - xyxy_to_xywh_norm: convert xyxy pixel boxes to normalized xywh for Ultralytics loss
    - build_ultra_batch: build the Ultralytics batch dict used by model.loss(...)
    - predict_packed: inference helper returning fixed-size packed outputs (boxes/labels/scores/valid)
    - predict_class_logits: logits-like (N,C) aggregation for KDD (from detection confidences)
    - predict_boxes_logits: (N,M,4), (N,M,C), (N,M) for BoxMatchKDD
    """

    def __init__(
        self,
        num_classes: int, imgsz: int, conf: float, iou: float,
        weights_path: str = "yolo11n.pt", max_det: int = 300, 
        agnostic_nms: bool = False, device: str | None = None
    ) -> None:
        super().__init__()

        self.yolo = YOLO(weights_path)

        self.num_classes = int(num_classes)
        self.imgsz = int(imgsz)
        self.conf = float(conf)
        self.iou = float(iou)
        self.max_det = int(max_det)
        self.agnostic_nms = bool(agnostic_nms)

        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self.to(self.device)

    def to(self, *args, **kwargs) -> YOLO11Detector:
        super().to(*args, **kwargs)
        self.yolo.to(*args, **kwargs)
        return self

    @staticmethod
    def xyxy_to_xywh_norm(boxes_xyxy: Tensor, h: int, w: int) -> Tensor:
        x1, y1, x2, y2 = boxes_xyxy.unbind(dim=1)
        bw = (x2 - x1).clamp(min=1.0)
        bh = (y2 - y1).clamp(min=1.0)
        cx = (x1 + x2) * 0.5
        cy = (y1 + y2) * 0.5
        return torch.stack([cx / float(w), cy / float(h), bw / float(w), bh / float(h)], dim=1).clamp(0.0, 1.0)

    def build_ultra_batch(self, x: Tensor, targets: list[dict[str, Tensor]]) -> dict[str, Tensor]:
        n, _, h, w = x.shape
        device = x.device

        batch_idx_all, cls_all, bboxes_all = [], [], []

        for i in range(n):
            t = targets[i]
            boxes, labels = t.get("boxes"), t.get("labels")
            if boxes is None or labels is None or boxes.numel() == 0 or labels.numel() == 0:
                continue

            boxes = boxes.to(device, torch.float32)
            labels = labels.to(device, torch.long)

            b_xywh = self.xyxy_to_xywh_norm(boxes, h, w)
            b_cls = labels.view(-1, 1).to(torch.float32)
            b_bi = torch.full((b_xywh.size(0),), i, device, torch.long)

            batch_idx_all.append(b_bi)
            cls_all.append(b_cls)
            bboxes_all.append(b_xywh)

        if not batch_idx_all:
            z0 = torch.zeros((0,), device, torch.long)
            zc = torch.zeros((0, 1), device, torch.float32)
            zb = torch.zeros((0, 4), device, torch.float32)
            return {"img": x, "batch_idx": z0, "cls": zc, "bboxes": zb}

        return {
            "img": x, "batch_idx": torch.cat(batch_idx_all, dim=0),
            "cls": torch.cat(cls_all, dim=0), "bboxes": torch.cat(bboxes_all, dim=0)}

    def forward(self, x: Tensor, targets: list[dict[str, Tensor]] | None) -> tuple[list[dict[str, Tensor]], dict[str, Tensor]]:
        if isinstance(x, (list, tuple)):
            x = torch.stack(list(x), dim=0)
        if x.device != self.device:
            x = x.to(self.device, non_blocking=True)

        if targets is not None:
            self.yolo.model.train()

            batch = self.build_ultra_batch(x, targets)
            preds = self.yolo.model(batch["img"])
            out = self.yolo.model.loss(preds, batch)

            loss_total, loss_items = (out[0], out[1]) if isinstance(out, (tuple, list)) else (out, None)
            loss_dict: dict[str, Tensor] = {"loss": loss_total, "total": loss_total}

            if torch.is_tensor(loss_items):
                items = loss_items.view(-1)

                if int(items.numel()) >= 3:
                    loss_dict["loss_box"] = items[0]
                    loss_dict["loss_cls"] = items[1]
                    loss_dict["loss_dfl"] = items[2]

                for j in range(int(items.numel())):
                    loss_dict[f"loss_item_{j}"] = items[j]

            return [], loss_dict

        self.yolo.model.eval()
        with torch.inference_mode():
            boxes_b, labels_b, scores_b, valid_b = self.predict_packed(x)

        outputs: list[dict[str, Tensor]] = []
        for i in range(x.shape[0]):
            v = valid_b[i]
            outputs.append({"boxes": boxes_b[i][v], "labels": labels_b[i][v], "scores": scores_b[i][v]})
        return outputs, {}

    @torch.inference_mode()
    def predict_packed(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        if x.device != self.device:
            x = x.to(self.device, non_blocking=True)

        results = self.yolo.predict(
            x, self.imgsz, self.conf, self.iou,
            self.max_det, self.agnostic_nms, self.device, verbose=False)

        n, m = x.shape[0], self.max_det
        boxes_b  = x.new_zeros((n, m, 4), torch.float32)
        labels_b = x.new_full((n, m), -1, torch.long)
        scores_b = x.new_zeros((n, m), torch.float32)
        valid_b  = x.new_zeros((n, m), torch.bool)

        for i, r in enumerate(results):
            b = getattr(r, "boxes", None)
            if b is None:
                continue

            xyxy = getattr(b, "xyxy", None)
            if xyxy is None:
                continue

            boxes = torch.as_tensor(xyxy, self.device, torch.float32)
            if boxes.numel() == 0:
                continue

            labels = torch.as_tensor(getattr(b, "cls", []), self.device, torch.long)
            scores = torch.as_tensor(getattr(b, "conf", []), self.device, torch.float32)

            k = min(m, int(boxes.shape[0]))
            boxes_b[i, :k]  = boxes[:k]
            labels_b[i, :k] = labels[:k]
            scores_b[i, :k] = scores[:k]
            valid_b[i, :k]  = True

        return boxes_b, labels_b, scores_b, valid_b

    @torch.no_grad()
    def predict_class_logits(self, x: Tensor) -> Tensor:
        if isinstance(x, (list, tuple)):
            x = torch.stack(list(x), dim=0)
        if x.device != self.device:
            x = x.to(self.device, non_blocking=True)

        self.eval()
        _, labels_b, scores_b, valid_b = self.predict_packed(x)

        n, c = x.size(0), int(self.num_classes)
        agg = x.new_zeros((n, c), torch.float32)

        for i in range(n):
            v = valid_b[i]
            if not bool(v.any()):
                continue

            labs = labels_b[i][v].clamp(0, c - 1)
            sc = scores_b[i][v].clamp(0.0, 1.0)
            for j in range(labs.numel()):
                agg[i, int(labs[j].item())] += sc[j]

        return torch.log(agg + EPS)

    def predict_boxes_logits(self, images: Tensor | list[Tensor]) -> tuple[Tensor, Tensor, Tensor]:
        x = torch.stack(images, 0) if isinstance(images, list) else images
        if x.device != self.device:
            x = x.to(self.device, non_blocking=True)

        boxes_b, labels_b, scores_b, valid_b = self.predict_packed(x)

        n, m = labels_b.shape
        c = int(self.num_classes)

        logits_b = x.new_full((n, m, c), -20.0, torch.float32)

        for i in range(n):
            v = valid_b[i]
            if not bool(v.any()):
                continue

            idx = torch.nonzero(v, as_tuple=False).view(-1)
            labs = labels_b[i][v].clamp(0, c - 1)
            sc = scores_b[i][v].clamp(0.0, 1.0)
            logits_b[i, idx, labs] = torch.log(sc + EPS)

        return boxes_b, logits_b, valid_b


def get_model_yolo11(cfg: ExperimentConfig) -> nn.Module:
    return YOLO11Detector(
        num_classes=cfg.data.num_classes, imgsz=cfg.data.img_size,
        weights_path=cfg.model.yolo_weights,conf=cfg.ssl.pseudo_conf_thr, iou=cfg.model.yolo_iou,
        max_det=cfg.model.yolo_max_det, agnostic_nms=cfg.model.yolo_agnostic_nms, device=cfg.train.device
    ).to(torch.device(cfg.train.device))
