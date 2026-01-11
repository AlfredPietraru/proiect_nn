from __future__ import annotations

from typing import Dict

import torch

from bbox.box_utils import box_iou
from .gradcam_train import GradCAMPP


@torch.no_grad()
def top1_acc(logits: torch.Tensor, y: torch.Tensor) -> float:
    return float((logits.argmax(dim=1) == y).float().mean().item())


def evaluate_cam_bboxes(
    campp: GradCAMPP,           # GradCAM++ model
    model: torch.nn.Module,     # same model used for logits
    images: torch.Tensor,       # (N,3,H,W)
    gt_boxes: torch.Tensor,     # (N,G,4)
    gt_labels: torch.Tensor,    # (N,G)
    iou_thr: float = 0.5, cam_thr: float = 0.35
) -> Dict[str, float]:
    N = images.shape[0]

    logits = model(images)
    main_gt = gt_labels[:, 0].clamp_min(0)
    acc = top1_acc(logits, main_gt)

    pred_boxes, pred_cls, _, pred_valid = campp(
        images, class_idx=main_gt, topk=1,
        threshold=cam_thr, use_gradients=True,
        detach_outputs=True)

    pred_boxes = pred_boxes[:, 0]    # (N,4)
    pred_cls = pred_cls[:, 0]        # (N,)
    pred_valid = pred_valid[:, 0]    # (N,)

    ious = []
    hit, gt_cnt, valid_cnt = 0, 0, 0

    for i in range(N):
        gmask = gt_labels[i] >= 0
        if not bool(gmask.any()):
            continue

        gtb = gt_boxes[i][gmask]   # (Gi,4)
        gtl = gt_labels[i][gmask]  # (Gi,)
        gt_cnt += 1

        if not bool(pred_valid[i]):
            continue
        valid_cnt += 1

        cmask = gtl == pred_cls[i]
        if not bool(cmask.any()):
            ious.append(pred_boxes.new_tensor(0.0))
            continue

        gtb_c = gtb[cmask]
        best_iou = box_iou(pred_boxes[i].view(1, 4), gtb_c).max().view(())
        ious.append(best_iou)

        if float(best_iou) >= float(iou_thr):
            hit += 1

    mean_iou = float(torch.stack(ious).mean().item()) if len(ious) else 0.0
    recall = hit / max(1, gt_cnt)
    hit_rate = hit / max(1, valid_cnt)
    valid_ratio = valid_cnt / max(1, gt_cnt)

    return {
        "acc_top1": float(acc),
        "mean_iou": float(mean_iou),
        f"recall@{iou_thr}": float(recall),
        f"hit_rate@{iou_thr}": float(hit_rate),
        "valid_ratio": float(valid_ratio),
    }
