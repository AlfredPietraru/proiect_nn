from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import torch

from bbox.box_ops import BoxList
from evaluate.ap import APStats, ap_for_class
from evaluate.pr import PRStats, pr_for_class


def stats_for_class(
    store: DetectionStore,
    selector: ClassSelector,
    cls: int, cfg: IoUMetrics
) -> Tuple[APStats, PRStats]:
    pred_boxes_list, pred_scores_list, tgt_boxes_list = [], [], []

    for pred_bl, tgt_bl in zip(store.preds, store.tgts):
        pred_boxes, pred_scores, tgt_boxes = selector.select(pred_bl, tgt_bl, cls)
        pred_boxes_list.append(pred_boxes)
        pred_scores_list.append(pred_scores)
        tgt_boxes_list.append(tgt_boxes)

    # Score threshold for predictions to be considered
    score_thr = float(cfg.score_thresh)

    # Average Precision (AP)
    ap_st = ap_for_class(
        pred_boxes_list, pred_scores_list,
        tgt_boxes_list, cfg.iou_thrs, score_thr)

    # Precision, Recall, F1
    pr_st = pr_for_class(
        pred_boxes_list, pred_scores_list,
        tgt_boxes_list, iou_thr=0.5, score_thr=score_thr)

    return ap_st, pr_st


@dataclass(frozen=True)
class IoUMetrics:
    num_classes: int
    class_agnostic: bool            # Whether to ignore class labels in evaluation
    iou_thrs: Tuple[float, ...]     # IoU thresholds for AP calculation
    score_thresh: float             # Score threshold for considering predictions

    def metric_names(self) -> List[str]:
        return [
            "mAP_50", "mAP_5095",
            "precision", "recall", "f1",
            "ap_num_pred", "ap_num_gt",
            "pr_num_pred", "pr_num_gt"
        ]


@dataclass
class DetectionStore:
    preds: List[BoxList]
    tgts: List[BoxList]

    def __init__(self) -> None:
        self.preds = []
        self.tgts = []

    def reset(self) -> None:
        self.preds.clear()
        self.tgts.clear()

    def update(self, preds: List[BoxList], tgts: List[BoxList]) -> None:
        self.preds.extend(preds)
        self.tgts.extend(tgts)


@dataclass(frozen=True)
class ClassSelector:
    cfg: IoUMetrics

    def num_classes(self) -> int:
        return 1 if self.cfg.class_agnostic else self.cfg.num_classes

    def select(
        self,
        pred_bl: BoxList,
        tgt_bl: BoxList,
        cls: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.cfg.class_agnostic:
            pred_mask = torch.ones_like(pred_bl.labels, dtype=torch.bool)
            tgt_mask = torch.ones_like(tgt_bl.labels, dtype=torch.bool)
        else:
            pred_mask = pred_bl.labels == cls
            tgt_mask = tgt_bl.labels == cls

        pred_boxes = pred_bl.boxes[pred_mask]
        if pred_bl.scores is not None:
            pred_scores = pred_bl.scores[pred_mask]
        else:
            pred_scores = torch.zeros((int(pred_mask.sum().item()),), device=pred_bl.boxes.device)

        tgt_boxes = tgt_bl.boxes[tgt_mask]
        return pred_boxes, pred_scores, tgt_boxes


class DetectionMetrics:
    def __init__(self, cfg: IoUMetrics):
        self.cfg = cfg
        self.store = DetectionStore()
        self.selector = ClassSelector(cfg)
        # Caching the metrics for - AP and PR per class - to avoid recomputation
        self.cache_metrics: Optional[Tuple[Dict[int, APStats], Dict[int, PRStats]]] = None

    def reset(self) -> None:
        self.store.reset()
        self.cache_metrics = None

    def update(self, preds: List[BoxList], targets: List[BoxList]) -> None:
        self.store.update(preds, targets)
        self.cache_metrics = None

    def core(self) -> Tuple[Dict[int, APStats], Dict[int, PRStats]]:
        if self.cache_metrics is not None:
            return self.cache_metrics

        num_classes = self.selector.num_classes()
        ap_per_class: Dict[int, APStats] = {}
        pr_per_class: Dict[int, PRStats] = {}

        for cls in range(num_classes):
            ap_st, pr_st = stats_for_class(self.store, self.selector, cls, self.cfg)
            ap_per_class[cls] = ap_st
            pr_per_class[cls] = pr_st

        self.cache_metrics = (ap_per_class, pr_per_class)
        return self.cache_metrics

    def compute(self) -> Dict[str, float]:
        ap_per_class, pr_per_class = self.core()
        num_classes = self.selector.num_classes()
        thrs = self.cfg.iou_thrs

        # Average Precision (AP)
        mAP_50 = 0.0
        if 0.5 in thrs:
            mAP_50 = sum((ap_per_class[c].values or {}).get(0.5, 0.0) for c in range(num_classes)) / max(1, num_classes)

        all_ap = []
        for c in range(num_classes):
            ap_dict = ap_per_class[c].values or {}
            for t in thrs:
                all_ap.append(ap_dict.get(t, 0.0))
        mAP_5095 = sum(all_ap) / max(1, len(all_ap))

        # Precision, Recall, F1
        p = sum((pr_per_class[c].values or {})["precision"] for c in range(num_classes)) / max(1, num_classes)
        r = sum((pr_per_class[c].values or {})["recall"] for c in range(num_classes)) / max(1, num_classes)
        f1 = sum((pr_per_class[c].values or {})["f1"] for c in range(num_classes)) / max(1, num_classes)

        # Count the number of predictions and ground truths
        ap_num_pred = sum(ap_per_class[c].num_pred for c in range(num_classes))
        ap_num_gt = sum(ap_per_class[c].num_gt for c in range(num_classes))

        pr_num_pred = sum(pr_per_class[c].num_pred for c in range(num_classes))
        pr_num_gt = sum(pr_per_class[c].num_gt for c in range(num_classes))

        return {
            "mAP_50": float(mAP_50), "mAP_5095": float(mAP_5095),
            "precision": float(p), "recall": float(r), "f1": float(f1),
            "ap_num_pred": float(ap_num_pred), "ap_num_gt": float(ap_num_gt),
            "pr_num_pred": float(pr_num_pred), "pr_num_gt": float(pr_num_gt)
        }
