from __future__ import annotations

from dataclasses import dataclass

import torch

from bbox.box_ops import box_iou

EPS: float = 1e-6

@dataclass
class APStats:
    num_pred: int = 0
    num_gt: int = 0
    values: dict[float, float] | None = None  # avg precision per IoU threshold

    def set(self, values: dict[float, float]) -> None:
        """Set the average precision values for different IoU thresholds."""
        self.values = values

    @staticmethod
    def avg_prec(recall: torch.Tensor, precision: torch.Tensor) -> float:
        """Calculate average precision given recall and precision tensors."""
        if recall.numel() == 0:
            return 0.0

        order = torch.argsort(recall)
        recall = recall[order]
        precision = precision[order]

        zero = recall.new_tensor([0.0])
        one = recall.new_tensor([1.0])

        mrec = torch.cat([zero, recall, one])
        mpre = torch.cat([zero, precision, zero])

        for i in range(mpre.numel() - 1, 0, -1):
            mpre[i - 1] = torch.maximum(mpre[i - 1], mpre[i])

        change = (mrec[1:] != mrec[:-1]).nonzero(as_tuple=False).squeeze(1)
        ap = torch.sum((mrec[change + 1] - mrec[change]) * mpre[change + 1])
        return ap.item()


def collect_matches_multi_thr(
    pred_boxes_list: list[torch.Tensor],
    pred_scores_list: list[torch.Tensor],
    tgt_boxes_list: list[torch.Tensor],
    iou_thrs: tuple[float, ...], score_thr: float,
    st: APStats
) -> tuple[torch.Tensor, dict[float, list[int]]]:
    """Collect matches between predicted and target boxes for multiple IoU thresholds."""
    all_scores: list[torch.Tensor] = []
    all_match: dict[float, list[int]] = {t: [] for t in iou_thrs}
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
            for t in iou_thrs:
                all_match[t].extend([0] * n_pred)
            continue

        ious = box_iou(pred_boxes, tgt_boxes)

        for thr in iou_thrs:
            used_gt = set()
            m_img: list[int] = []

            for pi in range(n_pred):
                max_iou, gi = ious[pi].max(dim=0)
                gi = int(gi.item())
                if max_iou >= thr and gi not in used_gt:
                    m_img.append(1)
                    used_gt.add(gi)
                else:
                    m_img.append(0)

            all_match[thr].extend(m_img)

    if len(all_scores) == 0:
        return torch.empty((0,), dtype=torch.float32), all_match
    return torch.cat(all_scores, dim=0), all_match


def ap_for_class(
    pred_boxes_list: list[torch.Tensor],
    pred_scores_list: list[torch.Tensor],
    tgt_boxes_list: list[torch.Tensor],
    iou_thrs: tuple[float, ...], score_thr: float
) -> APStats:
    """Calculate average precision for a specific class over multiple IoU thresholds."""
    st = APStats(values=None)

    # Collect matches for all IoU thresholds at once
    scores, all_match = collect_matches_multi_thr(
        pred_boxes_list, pred_scores_list, tgt_boxes_list,
        iou_thrs=iou_thrs, score_thr=score_thr, st=st)

    # Calculate AP per IoU threshold
    out = {t: 0.0 for t in iou_thrs}
    if st.num_pred == 0 or st.num_gt == 0 or scores.numel() == 0:
        st.set(out)
        return st

    # Sort scores in descending order
    order = scores.argsort(descending=True)

    # For each IoU threshold calculate AP
    for thr in iou_thrs:
        # Cumulate true positives and false positives
        m = torch.tensor(all_match[thr], dtype=torch.float32, device=scores.device)[order]

        # Over current IoU threshold
        cum_tp = torch.cumsum(m, dim=0)
        cum_fp = torch.cumsum(1.0 - m, dim=0)

        # Calculate precision and recall over all predictions
        den = (cum_tp + cum_fp)
        precision = torch.where(den > 0, cum_tp / den, den.new_zeros(()).expand_as(den))
        recall = cum_tp / float(st.num_gt) if st.num_gt > 0 else cum_tp.new_zeros(cum_tp.shape)
        out[thr] = float(APStats.avg_prec(recall, precision))

    st.set(out)
    return st
