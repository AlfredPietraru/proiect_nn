from __future__ import annotations

from typing import Tuple

from ap import APStats, ap_for_class
from pr import PRStats, pr_for_class
from .metrics import DetectionStore, Metrics, ClassSelector


def stats_for_class(
    store: DetectionStore,
    selector: ClassSelector,
    cls: int, cfg: Metrics
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
