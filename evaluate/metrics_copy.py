import torch
from functools import lru_cache

from evaluate.metrics import Metrics
from bbox.box_ops import BoxList, box_iou


def avg_prec(recall: torch.Tensor, precision: torch.Tensor) -> float:
    """
    Average Precision (AP):
    - area under the P-R curve.
    - using a number of interpolation points.
    - it is assumed that recall and precision are 1D tensors of the same length.
    (the scores associated with the points are not needed for AP computation)
    """
    order = torch.argsort(recall)
    recall = recall[order]
    precision = precision[order]

    mrec = torch.cat([torch.tensor([0.0]), recall, torch.tensor([1.0])])
    mpre = torch.cat([torch.tensor([0.0]), precision, torch.tensor([0.0])])

    for i in range(mpre.numel() - 1, 0, -1):
        mpre[i - 1] = torch.maximum(mpre[i - 1], mpre[i])

    idx = (mrec[1:] != mrec[:-1]).nonzero(as_tuple=False).squeeze()
    ap = torch.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])
    return float(ap)


class DetectionMetrics:
    def __init__(self, cfg: Metrics):
        self.cfg = cfg
        self.preds: list[BoxList] = []
        self.tgts: list[BoxList] = []
        self.cache: tuple[dict[int, dict[float, float]], dict[int, torch.Tensor]] | None = None

    def reset(self) -> None:
        self.preds.clear()
        self.tgts.clear()
        self.cache = None

    def update(self, preds: list[BoxList], targets: list[BoxList]) -> None:
        self.preds.extend(preds)
        self.tgts.extend(targets)
        self.cache = None

    @lru_cache(maxsize=None)
    def get_core(self) -> tuple[dict[int, dict[float, float]], dict[int, torch.Tensor]]:
        if self.cache is None:
            self.cache = self.metrics_core()
        return self.cache

    def num_classes(self) -> int:
        return 1 if self.cfg.class_agnostic else self.cfg.num_classes

    def init_ap_containers(self, num_classes: int) -> dict[int, dict[float, float]]:
        return {c: {t: 0.0 for t in self.cfg.iou_thrs} for c in range(num_classes)}

    def init_pr_stats(self, num_classes: int) -> dict[int, dict[str, float]]:
        return {c: {"tp": 0.0, "fp": 0.0, "fn": 0.0} for c in range(num_classes)}


    def select_boxes_for_class(
        self,
        pred_bl: BoxList,
        tgt_bl: BoxList,
        cls: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.cfg.class_agnostic:
            pred_mask = torch.ones_like(pred_bl.labels, dtype=torch.bool)
            tgt_mask = torch.ones_like(tgt_bl.labels, dtype=torch.bool)
        else:
            pred_mask = pred_bl.labels == cls
            tgt_mask = tgt_bl.labels == cls

        pred_boxes = pred_bl.boxes[pred_mask]
        pred_scores = pred_bl.scores[pred_mask]
        tgt_boxes = tgt_bl.boxes[tgt_mask]
        return pred_boxes, pred_scores, tgt_boxes

    def update_pr_stats_for_image(
        self,
        pred_boxes: torch.Tensor,
        pred_scores: torch.Tensor,
        tgt_boxes: torch.Tensor,
        pr_stats_cls: dict[str, float],
        num_gt: dict[float, int],
        iou_threshold: float = 0.5
    ):
        if iou_threshold not in self.cfg.iou_thrs:
            return
        if self.cfg.score_thresh is None:
            return

        num_gt[iou_threshold] += int(tgt_boxes.size(0))

        # no predictions at all
        if pred_boxes.numel() == 0:
            pr_stats_cls["fn"] += float(tgt_boxes.size(0))
            return

        # filter by score threshold
        sel = pred_scores >= self.cfg.score_thresh
        if sel.sum() == 0:
            pr_stats_cls["fn"] += float(tgt_boxes.size(0))
            return

        pred_sel = pred_boxes[sel]
        ious_sel = box_iou(pred_sel, tgt_boxes) if tgt_boxes.numel() > 0 else None
        used_gt = set()
        tp_count = 0
        fp_count = 0

        if ious_sel is not None:
            for pi in range(pred_sel.size(0)):
                max_iou, gi = ious_sel[pi].max(dim=0)
                gi_int = int(gi.item())
                if max_iou >= iou_threshold and gi_int not in used_gt:
                    tp_count += 1
                    used_gt.add(gi_int)
                else:
                    fp_count += 1

        pr_stats_cls["tp"] += float(tp_count)
        pr_stats_cls["fp"] += float(fp_count)
        pr_stats_cls["fn"] += float(max(0, tgt_boxes.size(0) - tp_count))

    def update_ap_bookkeeping_for_image(
        self,
        pred_boxes: torch.Tensor,
        pred_scores: torch.Tensor,
        tgt_boxes: torch.Tensor,
        iou_thresholds: tuple[float, ...],
        scores_all: list[float],
        matches_all: dict[float, list[int]],
        num_gt: dict[float, int],
    ):
        if tgt_boxes.numel() == 0 and pred_boxes.numel() == 0:
            return

        if tgt_boxes.numel() > 0:
            for thr in iou_thresholds:
                num_gt[thr] += int(tgt_boxes.size(0))

        if pred_boxes.numel() == 0:
            return

        scores_all.extend(pred_scores.tolist())

        if tgt_boxes.numel() > 0:
            ious = box_iou(pred_boxes, tgt_boxes)
            for thr in iou_thresholds:
                used_gt = set()
                matches_thr: list[int] = []
                for pi in range(pred_boxes.size(0)):
                    max_iou, gi = ious[pi].max(dim=0)
                    gi_int = int(gi.item())
                    if max_iou >= thr and gi_int not in used_gt:
                        matches_thr.append(1)  # TP
                        used_gt.add(gi_int)
                    else:
                        matches_thr.append(0)  # FP
                matches_all[thr].extend(matches_thr)
        else:
            for thr in iou_thresholds:
                matches_all[thr].extend([0] * pred_boxes.size(0))

    def ap_per_class(
        self,
        scores_all: list[float],
        matches_all: dict[float, list[int]],
        num_gt: dict[float, int],
        iou_thresholds: tuple[float, ...],
    ) -> dict[float, float]:
        AP_for_class: dict[float, float] = {t: 0.0 for t in iou_thresholds}

        # no predictions for this class at all
        if len(scores_all) == 0:
            return AP_for_class

        scores_tensor = torch.tensor(scores_all)
        order_global = scores_tensor.argsort(descending=True)

        for thr in iou_thresholds:
            if num_gt[thr] == 0:
                AP_for_class[thr] = 0.0
                continue

            matches_thr_tensor = torch.tensor(matches_all[thr])[order_global]
            tps = matches_thr_tensor
            fps = 1 - matches_thr_tensor

            cum_tp = torch.cumsum(tps, dim=0)
            cum_fp = torch.cumsum(fps, dim=0)

            recalls = cum_tp / float(num_gt[thr])
            precisions = cum_tp / (cum_tp + cum_fp + 1e-8)
            AP_for_class[thr] = float(avg_prec(recalls, precisions))

        return AP_for_class

    def accumulate_class_statistics(
        self,
        cls: int,
    ) -> tuple[
        dict[str, float],            # pr_stats_cls
        list[float],                 # scores_all
        dict[float, list[int]],      # matches_all
        dict[float, int],            # num_gt
    ]:
        iou_thresholds = self.cfg.iou_thrs

        # global accumulators for AP
        scores_all: list[float] = []
        matches_all: dict[float, list[int]] = {t: [] for t in iou_thresholds}
        num_gt: dict[float, int] = {t: 0 for t in iou_thresholds}

        # per-class PR stats
        pr_stats_cls: dict[str, float] = {"tp": 0.0, "fp": 0.0, "fn": 0.0}

        for pred_bl, tgt_bl in zip(self.preds, self.tgts):
            pred_boxes, pred_scores, tgt_boxes = self.select_boxes_for_class(
                pred_bl, tgt_bl, cls
            )

            # sort predictions by score (if any)
            if pred_boxes.numel() > 0:
                order = pred_scores.argsort(descending=True)
                pred_boxes = pred_boxes[order]
                pred_scores = pred_scores[order]

            # PR/F1 bookkeeping at IoU=0.5
            self.update_pr_stats_for_image(
                pred_boxes, pred_scores, tgt_boxes,
                pr_stats_cls, num_gt
            )

            # AP bookkeeping
            self.update_ap_bookkeeping_for_image(
                pred_boxes, pred_scores, tgt_boxes,
                iou_thresholds,
                scores_all, matches_all, num_gt
            )

        return pr_stats_cls, scores_all, matches_all, num_gt

    def prec_rec_f1(
        self,
        pr_stats: dict[int, dict[str, float]],
    ) -> tuple[dict[int, float], dict[int, float], dict[int, float]]:
        precision_per_class: dict[int, float] = {}
        recall_per_class: dict[int, float] = {}
        f1_per_class: dict[int, float] = {}

        num_classes = self.num_classes()

        for c in range(num_classes):
            tp = pr_stats[c]["tp"]
            fp = pr_stats[c]["fp"]
            fn = pr_stats[c]["fn"]

            prec = tp / (tp + fp + 1e-8) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn + 1e-8) if (tp + fn) > 0 else 0.0
            f1 = 2 * prec * rec / (prec + rec + 1e-8) if (prec + rec) > 0 else 0.0

            precision_per_class[c] = prec
            recall_per_class[c] = rec
            f1_per_class[c] = f1

        return precision_per_class, recall_per_class, f1_per_class

    def metrics_core(self) -> tuple[dict[int, dict[float, float]], dict[int, dict[str, float]]]:
        if len(self.preds) == 0:
            raise RuntimeError("No data added. Call update() first.")

        num_classes = self.num_classes()
        iou_thresholds = self.cfg.iou_thrs

        # Containers: per-class, per-threshold AP
        avg_per_per_cls: dict[int, dict[float, float]] = self.init_ap_containers(num_classes)
        # Simple precision/recall stats at IoU=0.5 and score_thresh
        pr_stats: dict[int, dict[str, float]] = self.init_pr_stats(num_classes)

        for cls in range(num_classes):
            pr_stats_cls, scores_all, matches_all, num_gt = self.accumulate_class_statistics(cls)
            AP_for_class = self.ap_per_class(scores_all, matches_all, num_gt, iou_thresholds)

            avg_per_per_cls[cls] = AP_for_class
            pr_stats[cls] = pr_stats_cls

        return avg_per_per_cls, pr_stats

    def compute_ap_per_class(self) -> dict[int, dict[float, float]]:
        AP_per_class, _ = self.get_core()
        return AP_per_class

    def compute_map_50(self) -> float:
        AP_per_class, _ = self.get_core()
        num_classes = self.num_classes()

        if 0.5 not in self.cfg.iou_thrs:
            return 0.0

        ap_50 = [AP_per_class[c][0.5] for c in range(num_classes)]
        return sum(ap_50) / max(1, len(ap_50))

    def compute_map_5095(self) -> float:
        AP_per_class, _ = self.get_core()
        num_classes = self.num_classes()
        iou_thresholds = self.cfg.iou_thrs

        ap_all = []
        for c in range(num_classes):
            for thr in iou_thresholds:
                ap_all.append(AP_per_class[c][thr])

        return sum(ap_all) / max(1, len(ap_all))

    def compute_precision(self) -> dict[int, float]:
        _, pr_stats = self.get_core()
        precision, _, _ = self.prec_rec_f1(pr_stats)
        return precision

    def compute_recall(self) -> dict[int, float]:
        _, pr_stats = self.get_core()
        _, recall, _ = self.prec_rec_f1(pr_stats)
        return recall

    def compute_f1(self) -> dict[int, float]:
        _, pr_stats = self.get_core()
        _, _, f1 = self.prec_rec_f1(pr_stats)
        return f1

    def compute(self) -> dict[str, float]:
        avg_per_per_cls, pr_stats = self.metrics_core()
        num_classes = self.num_classes()
        iou_thresholds = self.cfg.iou_thrs

        # mAP@0.5
        if 0.5 in iou_thresholds:
            ap_50 = [avg_per_per_cls[c][0.5] for c in range(num_classes)]
            mAP_50 = sum(ap_50) / max(1, len(ap_50))
        else:
            mAP_50 = 0.0

        # mAP@[0.5:0.95]
        ap_all = []
        for c in range(num_classes):
            for thr in iou_thresholds:
                ap_all.append(avg_per_per_cls[c][thr])
        mAP_5095 = sum(ap_all) / max(1, len(ap_all))

        # reuse shared precision/recall/F1 computation
        precision_per_class, recall_per_class, f1_per_class = self.prec_rec_f1(pr_stats)
        final_precision = sum(precision_per_class.values()) / max(1, len(precision_per_class.values()))
        final_recall = sum(recall_per_class.values()) / max(1, len(recall_per_class.values()))
        final_f1 = sum(f1_per_class.values()) / max(1, len(f1_per_class.values()))
        return {
            # "avg_per_per_cls": avg_per_per_cls,
            "mAP_50": mAP_50,
            "mAP_5095": mAP_5095,
            "precision": final_precision,
            "recall": final_recall,
            "f1": final_f1,
        }
    