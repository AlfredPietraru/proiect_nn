from __future__ import annotations

from dataclasses import dataclass
import torch

from bbox.box_ops import box_iou

EPS: float = 1e-6

@dataclass
class PRStats:
    num_pred: int = 0
    num_gt: int = 0
    values: dict[str, float] | None = None  # Precision, Recall, F1

    def set(self, values: dict[str, float]) -> None:
        self.values = values


def collect_matches_single_thr(
    pred_boxes_list: list[torch.Tensor],
    pred_scores_list: list[torch.Tensor],
    tgt_boxes_list: list[torch.Tensor],
    iou_thr: float, score_thr: float, st: PRStats
) -> tuple[torch.Tensor, list[int]]:
    """Collect matches between predicted and target boxes for a single IoU threshold."""
    all_scores: list[torch.Tensor] = []
    all_match: list[int] = []

    for pred_boxes, pred_scores, tgt_boxes in zip(pred_boxes_list, pred_scores_list, tgt_boxes_list):
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

    if len(all_scores) == 0:
        return torch.empty((0,), dtype=torch.float32), all_match
    return torch.cat(all_scores, dim=0), all_match


def pr_for_class(
    pred_boxes_list: list[torch.Tensor],
    pred_scores_list: list[torch.Tensor],
    tgt_boxes_list: list[torch.Tensor],
    score_thr: float, iou_thr: float = 0.5
) -> PRStats:
    """Calculate precision, recall, and F1 score for a specific class."""
    st = PRStats(values=None)

    # Collect matches for the given IoU threshold
    scores, match = collect_matches_single_thr(
        pred_boxes_list, pred_scores_list, tgt_boxes_list,
        iou_thr=iou_thr, score_thr=score_thr, st=st)

    if st.num_pred == 0:
        st.set({"precision": 0.0, "recall": 0.0, "f1": 0.0})
        return st

    # Cumulate true positives and false positives
    m = torch.tensor(match, dtype=torch.float32, device=scores.device)

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
