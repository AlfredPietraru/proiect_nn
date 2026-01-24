from __future__ import annotations

import torch
from torch import Tensor

EPS: float = 1e-6

def box_area(boxes: Tensor) -> Tensor:
    wh = (boxes[:, 2:] - boxes[:, :2]).clamp(min=0)
    return wh[:, 0] * wh[:, 1]


def box_iou(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return boxes1.new_zeros((boxes1.size(0), boxes2.size(0)))

    b1 = boxes1[:, None, :]  # (N,1,4)
    b2 = boxes2[None, :, :]  # (1,M,4)

    x1 = torch.maximum(b1[..., 0], b2[..., 0])
    y1 = torch.maximum(b1[..., 1], b2[..., 1])
    x2 = torch.minimum(b1[..., 2], b2[..., 2])
    y2 = torch.minimum(b1[..., 3], b2[..., 3])

    inter_w = (x2 - x1).clamp(min=0)
    inter_h = (y2 - y1).clamp(min=0)
    inter = inter_w * inter_h

    area1 = ((b1[..., 2] - b1[..., 0]).clamp(min=0) * (b1[..., 3] - b1[..., 1]).clamp(min=0))
    area2 = ((b2[..., 2] - b2[..., 0]).clamp(min=0) * (b2[..., 3] - b2[..., 1]).clamp(min=0))

    union = area1 + area2 - inter
    return inter / union.clamp(min=1e-12)
