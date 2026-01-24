from __future__ import annotations

from dataclasses import dataclass
import torch
from torch import Tensor

from .box_utils import box_iou
from .boxes import AnchorBoxes

EPS: float = 1e-6

@dataclass
class BoxList:
    boxes: Tensor
    labels: Tensor
    scores: Tensor | None
    image_size: tuple[int, int]

    def to(self, device) -> BoxList:
        return BoxList(
            image_size=self.image_size, boxes=self.boxes.to(device), 
            labels=self.labels.to(device), scores=self.scores.to(device) if self.scores is not None else None)


def encode_boxes(
    anchors: Tensor,          # (N,4)
    gt_boxes: Tensor,         # (N,4)
    weights: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
) -> Tensor:
    """Encode ground-truth boxes with respect to anchors."""
    wx, wy, ww, wh = weights

    ax = (anchors[:, 0] + anchors[:, 2]) * 0.5
    ay = (anchors[:, 1] + anchors[:, 3]) * 0.5
    aw = anchors[:, 2] - anchors[:, 0]
    ah = anchors[:, 3] - anchors[:, 1]

    gx = (gt_boxes[:, 0] + gt_boxes[:, 2]) * 0.5
    gy = (gt_boxes[:, 1] + gt_boxes[:, 3]) * 0.5
    gw = gt_boxes[:, 2] - gt_boxes[:, 0]
    gh = gt_boxes[:, 3] - gt_boxes[:, 1]

    dx = wx * (gx - ax) / (aw + EPS)
    dy = wy * (gy - ay) / (ah + EPS)
    dw = ww * torch.log(gw / (aw + EPS) + EPS)
    dh = wh * torch.log(gh / (ah + EPS) + EPS)

    return torch.stack((dx, dy, dw, dh), dim=1)


def decode_boxes(
    anchors: Tensor,          # (N,4)
    deltas: Tensor,           # (N,4)
    weights: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
) -> Tensor:
    """Decode box regression deltas to boxes."""
    wx, wy, ww, wh = weights

    ax = (anchors[:, 0] + anchors[:, 2]) * 0.5
    ay = (anchors[:, 1] + anchors[:, 3]) * 0.5
    aw = anchors[:, 2] - anchors[:, 0]
    ah = anchors[:, 3] - anchors[:, 1]

    dx = deltas[:, 0] / wx
    dy = deltas[:, 1] / wy
    dw = deltas[:, 2] / ww
    dh = deltas[:, 3] / wh

    px = dx * aw + ax
    py = dy * ah + ay
    pw = torch.exp(dw) * aw
    ph = torch.exp(dh) * ah

    x1 = px - 0.5 * pw
    y1 = py - 0.5 * ph
    x2 = px + 0.5 * pw
    y2 = py + 0.5 * ph
    return torch.stack((x1, y1, x2, y2), dim=1)


def anchors_to_boxlist(
    anchors: AnchorBoxes | Tensor,
    scores: Tensor, labels: Tensor,
    image_size: tuple[int, int] | None = None
) -> BoxList:
    """Convert anchors and associated scores and labels to a BoxList."""
    if torch.is_tensor(anchors):
        if image_size is None:
            raise ValueError("Image size is required when boxes is a tensor")
        boxes = anchors
        imsz = image_size
    else:
        boxes = anchors.boxes
        imsz = anchors.image_size

    return BoxList(boxes=boxes, scores=scores, labels=labels, image_size=imsz)


def filter_by_image_labels(boxlist: BoxList, image_labels: Tensor) -> BoxList:
    """Filter BoxList to keep only boxes whose labels are present in image_labels."""
    present = torch.nonzero(image_labels > 0).squeeze(1)
    mask = torch.isin(boxlist.labels, present.to(boxlist.labels.device))
    return BoxList(
        image_size=boxlist.image_size, boxes=boxlist.boxes[mask], labels=boxlist.labels[mask],
        scores=boxlist.scores[mask] if boxlist.scores is not None else None)


def match_anchors_to_gt(
    anchors: Tensor, gt: BoxList,  # (Na,4), (Ng,4)
    high_thresh: float = 0.95, low_thresh: float = 0.05,
    allow_low_quality_matches: bool = True
) -> tuple[Tensor, Tensor, Tensor]:
    """Match anchors to ground-truth boxes."""
    if gt.boxes.numel() == 0:
        raise ValueError("No ground-truth boxes provided for matching.")
    if gt.boxes.ndim != 2 or gt.boxes.size(1) != 4:
        raise ValueError("For GT must be shaped (Ng,4) boxes.")

    ious = box_iou(anchors, gt.boxes)              # (Na, Ng)
    max_iou, matched = ious.max(dim=1)             # (Na,)

    neg = max_iou < low_thresh
    pos = max_iou >= high_thresh

    if allow_low_quality_matches:
        best_anchor_per_gt = ious.argmax(dim=0)    # (Ng,)
        pos[best_anchor_per_gt] = True
        matched[best_anchor_per_gt] = torch.arange(gt.boxes.size(0), anchors.device)

    labels = torch.zeros((anchors.size(0),), torch.long, anchors.device)
    labels[pos] = gt.labels[matched[pos]]

    matched_gt_inds = matched.clone()
    matched_gt_inds[neg] = -1

    bbox_targets = torch.zeros((anchors.size(0), 4), anchors.device)
    if pos.any():
        bbox_targets[pos] = encode_boxes(anchors[pos], gt.boxes[matched[pos]])

    return labels, bbox_targets, matched_gt_inds


def aggregate_boxes(boxlist: BoxList, iou_merge_thresh: float = 0.5, min_group_size: int = 1) -> BoxList:
    """Aggregate boxes in BoxList based on IoU and scores."""
    boxes = boxlist.boxes
    labels = boxlist.labels
    scores = boxlist.scores

    if boxes.numel() == 0:
        return boxlist
    if scores is None:
        scores = torch.ones((boxes.size(0),), boxes.device, torch.float32)
    else:
        scores = scores.to(torch.float32)

    out_boxes, out_scores, out_labels = [], [], []

    for c in labels.unique():
        idx = torch.nonzero(labels == c).squeeze(1)
        if idx.numel() == 0:
            continue

        b = boxes[idx]
        s = scores[idx]

        order = torch.argsort(s, descending=True)
        b = b[order]
        s = s[order]

        kept = torch.ones((b.size(0),), torch.bool, b.device)

        for i in range(b.size(0)):
            if not kept[i]:
                continue

            seed = b[i].unsqueeze(0)                  # (1,4)
            ious = box_iou(seed, b)[0]                # (N,)
            group = (ious >= iou_merge_thresh) & kept

            group_idx = torch.nonzero(group).squeeze(1)
            if group_idx.numel() < min_group_size:
                kept[i] = False
                continue

            gb = b[group_idx]                         # (K,4)
            gs = s[group_idx]                         # (K,)

            w = gs / (gs.sum() + EPS)                 # normalized weights
            fused = (gb * w[:, None]).sum(dim=0)      # (4,)

            out_boxes.append(fused)
            out_scores.append(gs.mean())
            out_labels.append(c)

            kept[group_idx] = False

    if not out_boxes:
        raise ValueError("No boxes kept after aggregation.")

    out_boxes = torch.stack(out_boxes, dim=0)
    out_scores = torch.stack(out_scores, dim=0)
    out_labels = torch.stack(out_labels, dim=0)

    order = torch.argsort(out_scores, descending=True)
    return BoxList(boxes=out_boxes[order], scores=out_scores[order], labels=out_labels[order], image_size=boxlist.image_size)
