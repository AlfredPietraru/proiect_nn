from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from .box_utils import box_area
from .points import AnchorPoints

EPS = 1e-6


@dataclass
class PointList:
    points: torch.Tensor
    labels: torch.Tensor
    scores: Optional[torch.Tensor]
    image_size: Tuple[int, int]

    def to(self, device) -> "PointList":
        return PointList(
            points=self.points.to(device),
            labels=self.labels.to(device),
            scores=self.scores.to(device) if self.scores is not None else None,
            image_size=self.image_size,
        )


def encode_ltrb(points: torch.Tensor, gt_boxes: torch.Tensor) -> torch.Tensor:
    cx, cy = points[:, 0], points[:, 1]
    left = cx - gt_boxes[:, 0]
    top = cy - gt_boxes[:, 1]
    right = gt_boxes[:, 2] - cx
    bottom = gt_boxes[:, 3] - cy
    return torch.stack((left, top, right, bottom), dim=1)


def decode_ltrb(points: torch.Tensor, ltrb: torch.Tensor) -> torch.Tensor:
    cx, cy = points[:, 0], points[:, 1]
    x1 = cx - ltrb[:, 0]
    y1 = cy - ltrb[:, 1]
    x2 = cx + ltrb[:, 2]
    y2 = cy + ltrb[:, 3]
    return torch.stack((x1, y1, x2, y2), dim=1)


def points_to_pointlist(
    points: AnchorPoints | torch.Tensor,
    scores: torch.Tensor,
    labels: torch.Tensor,
    image_size: Optional[Tuple[int, int]] = None,
) -> PointList:
    if torch.is_tensor(points):
        if image_size is None:
            raise ValueError("Image size must be provided when points is a tensor.")
        pts = points
        imsz = image_size
    else:
        pts = points.points
        imsz = points.image_size

    return PointList(points=pts, scores=scores, labels=labels, image_size=imsz)


def filter_points_by_image_labels(pointlist: PointList, image_labels: torch.Tensor) -> PointList:
    present = torch.nonzero(image_labels > 0).squeeze(1)
    mask = torch.isin(pointlist.labels, present.to(pointlist.labels.device))
    return PointList(
        points=pointlist.points[mask],
        scores=pointlist.scores[mask] if pointlist.scores is not None else None,
        labels=pointlist.labels[mask],
        image_size=pointlist.image_size,
    )


def match_points_to_gt(
    points: torch.Tensor,   # (Np,2)
    gt: PointList,          # (Ng)
    allow_low_quality_matches: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if gt.points.numel() == 0:
        raise ValueError("No ground-truth boxes provided for matching.")
    if gt.points.ndim != 2 or gt.points.size(1) != 4:
        raise ValueError("For GT must be shaped (Ng,4) boxes (x1,y1,x2,y2).")

    gt_boxes = gt.points        # (Ng,4)
    gt_labels = gt.labels       # (Ng,)

    Np = points.size(0)
    Ng = gt_boxes.size(0)

    px = points[:, 0][:, None]  # (Np,1)
    py = points[:, 1][:, None]  # (Np,1)

    x1 = gt_boxes[None, :, 0]   # (1,Ng)
    y1 = gt_boxes[None, :, 1]   # (1,Ng)
    x2 = gt_boxes[None, :, 2]   # (1,Ng)
    y2 = gt_boxes[None, :, 3]   # (1,Ng)

    inside = (px >= x1) & (px <= x2) & (py >= y1) & (py <= y2)  # (Np,Ng)

    areas = box_area(gt_boxes).to(dtype=torch.float32)          # (Ng,)
    cost = areas[None, :].expand(Np, Ng).clone()
    cost[~inside] = float("inf")

    best_cost, matched = cost.min(dim=1)                        # (Np,)
    pos = torch.isfinite(best_cost)
    neg = ~pos

    if allow_low_quality_matches:
        gcx = (gt_boxes[:, 0] + gt_boxes[:, 2]) * 0.5           # (Ng,)
        gcy = (gt_boxes[:, 1] + gt_boxes[:, 3]) * 0.5
        dx = points[:, 0][:, None] - gcx[None, :]
        dy = points[:, 1][:, None] - gcy[None, :]
        dist2 = dx * dx + dy * dy                               # (Np,Ng)

        nearest_point_per_gt = dist2.argmin(dim=0)              # (Ng,)
        pos[nearest_point_per_gt] = True
        matched[nearest_point_per_gt] = torch.arange(Ng, device=points.device)

    labels = torch.zeros((Np,), dtype=torch.long, device=points.device)
    labels[pos] = gt_labels[matched[pos]]

    matched_inds = matched.clone()
    matched_inds[neg] = -1

    targets = torch.zeros((Np, 4), device=points.device)
    if pos.any():
        targets[pos] = encode_ltrb(points[pos], gt_boxes[matched[pos]])

    return labels, targets, matched_inds


def aggregate_points(
    pointlist: PointList,
    dist_thresh: float = 5.0,  # Distance threshold to consider points for merging
    min_group_size: int = 1,  # Minimum number of points in a group to perform merging
) -> PointList:
    pts = pointlist.points
    labels = pointlist.labels
    scores = pointlist.scores

    if pts.numel() == 0:
        return pointlist
    if scores is None:
        scores = torch.ones((pts.size(0),), device=pts.device, dtype=torch.float32)
    else:
        scores = scores.to(dtype=torch.float32)

    out_pts, out_scores, out_labels = [], [], []

    for c in labels.unique():
        idx = torch.nonzero(labels == c).squeeze(1)
        if idx.numel() == 0:
            continue

        p = pts[idx]
        s = scores[idx]

        order = torch.argsort(s, descending=True)
        p = p[order]
        s = s[order]

        alive = torch.ones((p.size(0),), dtype=torch.bool, device=p.device)

        for i in range(p.size(0)):
            if not alive[i]:
                continue

            d2 = ((p - p[i]) ** 2).sum(dim=1)
            group = (d2 <= dist_thresh * dist_thresh) & alive
            gidx = torch.nonzero(group).squeeze(1)

            if gidx.numel() < min_group_size:
                alive[i] = False
                continue

            gp = p[gidx]
            gs = s[gidx]

            w = gs / (gs.sum() + EPS)
            fused = (gp * w[:, None]).sum(dim=0)
            out_pts.append(fused)
            out_scores.append(gs.mean())
            out_labels.append(c)

            alive[gidx] = False

    if not out_pts:
        raise ValueError("No points kept after aggregation.")

    out_pts = torch.stack(out_pts, dim=0)
    out_scores = torch.stack(out_scores, dim=0)
    out_labels = torch.stack(out_labels, dim=0)

    order = torch.argsort(out_scores, descending=True)
    return PointList(
        points=out_pts[order],
        scores=out_scores[order],
        labels=out_labels[order],
        image_size=pointlist.image_size,
    )
