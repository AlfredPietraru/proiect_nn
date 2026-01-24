from __future__ import annotations

import torch
from torch import Tensor

from bbox.box_utils import box_iou
from .gradcam_train import GradCAMPP


@torch.no_grad()
def top1_acc(logits: Tensor, y: Tensor) -> float:
    return float((logits.argmax(dim=1) == y).float().mean().item())


def evaluate_cam_bboxes(
    campp: GradCAMPP, model: torch.nn.Module,
    images: Tensor, gt_boxes: Tensor, gt_labels: Tensor,
    iou_thr: float = 0.5, cam_thr: float = 0.35, top_k: int = 1
) -> dict[str, float]:
    """Evaluate CAM bounding boxes against ground truth boxes."""
    n = int(images.shape[0])

    logits = model(images)
    main_gt = gt_labels[:, 0].clamp_min(0)
    acc = top1_acc(logits, main_gt)

    with torch.enable_grad():
        x_cam = images.detach().requires_grad_(True)
        _, boxes, labels, _, valid = campp(x_cam, main_gt, top_k, cam_thr, use_gradients=True, detach_outputs=True)

    pred_boxes, pred_cls, pred_valid = boxes[:, 0, :], labels[:, 0], valid[:, 0]

    ious: list[Tensor] = []
    hit, gt_cnt, valid_cnt = 0, 0, 0

    for i in range(n):
        gmask = gt_labels[i] >= 0
        if not bool(gmask.any()):
            continue

        gtb, gtl = gt_boxes[i][gmask], gt_labels[i][gmask]
        gt_cnt += 1

        if not bool(pred_valid[i]):
            continue
        valid_cnt += 1

        cmask = gtl == pred_cls[i]
        if not bool(cmask.any()):
            ious.append(pred_boxes.new_tensor(0.0))
            continue

        best_iou = box_iou(pred_boxes[i].view(1, 4), gtb[cmask]).max().view(())
        ious.append(best_iou)
        hit += int(float(best_iou) >= float(iou_thr))

    mean_iou = float(torch.stack(ious).mean().item()) if ious else 0.0
    recall = hit / max(1, gt_cnt)
    hit_rate = hit / max(1, valid_cnt)
    valid_ratio = valid_cnt / max(1, gt_cnt)

    return {f"acc_top{top_k}": acc, "mean_iou": mean_iou, f"recall@{iou_thr}": recall, f"hit_rate@{iou_thr}": hit_rate, "valid_ratio": valid_ratio}
