from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import torch

from bbox.box_ops import box_iou

EPS = 1e-8


@dataclass
class PRStats:
    num_pred: int = 0
    num_gt: int = 0
    values: Optional[Dict[str, float]] = None  # Precision, Recall, F1

    def set(self, values: Dict[str, float]) -> None:
        self.values = values


def collect_matches_single_thr(
    pred_boxes_list: List[torch.Tensor],
    pred_scores_list: List[torch.Tensor],
    tgt_boxes_list: List[torch.Tensor],
    iou_thr: float, score_thr: float,
    st: PRStats,
) -> Tuple[torch.Tensor, List[int]]:
    all_scores: List[torch.Tensor] = []
    all_match: List[int] = []

    for pred_boxes, pred_scores, tgt_boxes in zip(
        pred_boxes_list, pred_scores_list, tgt_boxes_list
    ):
        st.num_gt += int(tgt_boxes.size(0))

        if pred_boxes.numel() == 0:
            continue

        keep = pred_scores >= score_thr
        if int(keep.sum().item()) == 0:
            continue

        pred_boxes = pred_boxes[keep]
        pred_scores = pred_scores[keep]

        order = pred_scores.argsort(descending=True)
        pred_boxes = pred_boxes[order]
        pred_scores = pred_scores[order]

        n_pred = int(pred_boxes.size(0))
        st.num_pred += n_pred
        all_scores.append(pred_scores)

        if tgt_boxes.numel() == 0:
            all_match.extend([0] * n_pred)
            continue

        ious = box_iou(pred_boxes, tgt_boxes)
        used_gt = set()

        for pi in range(n_pred):
            max_iou, gi = ious[pi].max(dim=0)
            gi = int(gi.item())
            if max_iou >= iou_thr and gi not in used_gt:
                all_match.append(1)
                used_gt.add(gi)
            else:
                all_match.append(0)

    return torch.cat(all_scores, dim=0), all_match


def pr_for_class(
    pred_boxes_list: List[torch.Tensor],
    pred_scores_list: List[torch.Tensor],
    tgt_boxes_list: List[torch.Tensor],
    score_thr: float, iou_thr: float = 0.5,
) -> PRStats:
    st = PRStats(values=None)

    # Collect matches for the given IoU threshold
    scores, match = collect_matches_single_thr(
        pred_boxes_list, pred_scores_list, tgt_boxes_list,
        iou_thr=iou_thr, score_thr=score_thr, st=st
    )

    if st.num_pred == 0:
        st.set({"precision": 0.0, "recall": 0.0, "f1": 0.0})
        return st

    # Cumulate true positives and false positives
    m = torch.tensor(match, torch.float32, scores.device)

    # Over all predictions
    tp = float(m.sum().item())
    fp = float((1.0 - m).sum().item())
    fn = max(0.0, float(st.num_gt) - tp)

    # Calculate precision, recall, F1 score for current IoU threshold
    precision = tp / (tp + fp + EPS)
    recall = tp / (tp + fn + EPS)
    f1 = (2.0 * precision * recall / (precision + recall + EPS))

    st.set({"precision": precision, "recall": recall, "f1": f1})
    return st
