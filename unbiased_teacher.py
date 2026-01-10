from __future__ import annotations

import os
from typing import Dict, List, Tuple

import torch
from torch.utils.data import DataLoader
from torchvision.ops import batched_nms
from tqdm import tqdm
from loguru import logger

from .utils.checkpoints import load_checkpoint, save_checkpoint
from .core import move_images_to_device, move_targets_to_device
from .model_factory import build_model
from .models.scheduler import build_scheduler
from .models.ema import EMA
from .models.early_stopping import EarlyStopping
from .models.hyperparams import ExperimentConfig
from .bbox.box_ops import BoxList
from .data.visualize.training_curves import TrainingCurvesSemiSupervised
from .evaluate.metrics import DetectionMetrics, Metrics


@torch.no_grad()
def generate_pseudo_labels(
    model: torch.nn.Module,
    images: List[torch.Tensor],
    device: torch.device,
    score_thr: float,
    nms_iou: float,
) -> List[dict]:
    model.eval()
    images = move_images_to_device(images, device)
    outputs, _ = model(images, None)

    pseudo: List[dict] = []
    for out in outputs:
        boxes, labels, scores = out["boxes"], out["labels"], out["scores"]

        if boxes.numel() == 0:
            pseudo.append({"boxes": boxes, "labels": labels, "scores": scores})
            continue

        keep = batched_nms(boxes, scores, labels, iou_threshold=nms_iou)
        boxes, labels, scores = boxes[keep], labels[keep], scores[keep]

        keep = scores > score_thr
        boxes, labels, scores = boxes[keep], labels[keep], scores[keep]
        pseudo.append({"boxes": boxes.detach(), "labels": labels.detach(), "scores": scores.detach()})

    return pseudo


def filter_nonempty_pseudo(pseudo: List[dict]) -> Tuple[List[dict], List[int]]:
    keep_idx = [i for i, t in enumerate(pseudo) if t["boxes"].numel() > 0]
    kept = [pseudo[i] for i in keep_idx]
    kept = [{"boxes": t["boxes"], "labels": t["labels"]} for t in kept]
    return kept, keep_idx


def train_semi_supervised_one_epoch(
    teacher: EMA,
    student: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    data: Dict[str, DataLoader],
    device: torch.device,
    max_iter: int,
    lambda_unsup: float,
    score_thr: float,
    nms_iou: float,
    metric_sup: List[str],
    metric_unsup: List[str],
) -> Tuple[Dict[str, float], Dict[str, float], float]:
    student.train()

    hist_sup = {k: 0.0 for k in metric_sup}
    hist_unsup = {k: 0.0 for k in metric_unsup}
    total_loss = 0.0
    steps = 0

    labeled_loader = data["train_burn_in_strong"]
    labeled_it = iter(labeled_loader)

    unlabeled_loader = data["train_weak"]

    for step_idx, (img_weak, img_strong) in enumerate(tqdm(unlabeled_loader, desc="SSL train")):
        if step_idx >= max_iter:
            break

        try:
            img_labeled, targets_labeled = next(labeled_it)
        except StopIteration:
            labeled_it = iter(labeled_loader)
            img_labeled, targets_labeled = next(labeled_it)

        pseudo = generate_pseudo_labels(
            model=teacher.ema, images=img_weak, device=device,
            score_thr=score_thr, nms_iou=nms_iou)

        pseudo_kept, keep_idx = filter_nonempty_pseudo(pseudo)
        if not keep_idx:
            continue

        img_strong = move_images_to_device(img_strong, device)
        img_strong = [img_strong[i] for i in keep_idx]
        pseudo_kept = move_targets_to_device(pseudo_kept, device)

        _, loss_u = student(img_strong, pseudo_kept)
        unsup_cls = loss_u["loss_classifier"] + loss_u["loss_objectness"]

        img_labeled = move_images_to_device(img_labeled, device)
        targets_labeled = move_targets_to_device(targets_labeled, device)

        _, loss_s = student(img_labeled, targets_labeled)
        sup = sum(loss_s.values())

        loss = sup + lambda_unsup * unsup_cls

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()

        teacher.update(student)

        for k in metric_unsup:
            if k in loss_u:
                hist_unsup[k] += float(loss_u[k].item())
        for k in metric_sup:
            if k in loss_s:
                hist_sup[k] += float(loss_s[k].item())

        total_loss += float(loss.item())
        steps += 1

    denom = max(1, steps)
    for k in hist_unsup:
        hist_unsup[k] /= denom
    for k in hist_sup:
        hist_sup[k] /= denom

    return hist_sup, hist_unsup, total_loss / denom


@torch.no_grad()
def validate_semi_supervised(
    student: torch.nn.Module,
    dt_test: DataLoader,
    device: torch.device,
    cfg_metrics: Metrics,
    max_iter: int,
) -> Tuple[Dict[str, float], float]:
    student.eval()

    metrics = DetectionMetrics(cfg_metrics)
    metrics.reset()

    loss_sum = 0.0
    steps = 0

    for step_idx, (images, targets) in enumerate(tqdm(dt_test, desc="Validation")):
        if step_idx >= max_iter:
            break

        images = move_images_to_device(images, device)
        targets = move_targets_to_device(targets, device)

        outputs, loss_dict = student(images, targets)
        if loss_dict:
            loss_sum += float(sum(loss_dict.values()).item())

        preds_bl: List[BoxList] = []
        tgts_bl: List[BoxList] = []

        for img, out, tgt in zip(images, outputs, targets):
            h, w = int(img.shape[-2]), int(img.shape[-1])
            size_hw = (h, w)

            out_boxes = out.get("boxes", torch.zeros((0, 4), device=device))
            out_labels = out.get("labels", torch.zeros((0,), dtype=torch.int64, device=device))
            out_scores = out.get("scores", None)

            tgt_boxes = tgt.get("boxes", torch.zeros((0, 4), device=device))
            tgt_labels = tgt.get("labels", torch.zeros((0,), dtype=torch.int64, device=device))
            tgt_scores = tgt.get("scores", None)

            preds_bl.append(BoxList(out_boxes, out_labels, out_scores, size_hw))
            tgts_bl.append(BoxList(tgt_boxes, tgt_labels, tgt_scores, size_hw))

        metrics.update(preds_bl, tgts_bl)
        steps += 1

    avg_loss = loss_sum / max(1, steps)
    return metrics.compute(), avg_loss


def pipeline_semi_supervised(
    cfg: ExperimentConfig,
    checkpoint_path: str,  # checkpoint from burn-in phase
    data: Dict[str, DataLoader],
    device: torch.device,
    metric_sup: List[str],
    metric_unsup: List[str],
    eval_metrics: List[str],
) -> None:
    student = build_model(cfg=cfg).to(device)
    student, _, _ = load_checkpoint(path=checkpoint_path, model=student, optimizer=None, device=device)

    teacher = EMA(student, decay=cfg.ssl.ema_decay)
    optimizer = torch.optim.SGD(
        student.parameters(), lr=cfg.optim.lr, momentum=cfg.optim.momentum,
        weight_decay=cfg.optim.weight_decay, nesterov=cfg.optim.nesterov)

    steps_per_epoch = len(data["train_weak"])
    lr_scheduler = build_scheduler(
        optimizer=optimizer,
        scheme=cfg.sched.scheme, total_epochs=cfg.train.epochs, steps_per_epoch=steps_per_epoch,
        warmup_epochs=cfg.sched.warmup_epochs, warmup_bias_lr=cfg.sched.warmup_bias_lr,
        min_lr_ratio=cfg.sched.min_lr_ratio, milestones=cfg.sched.milestones, gamma=cfg.sched.gamma)

    early_stopping = EarlyStopping(patience=8, min_delta=1e-3, mode="min", verbose=True)

    cfg_metrics = Metrics(
        num_classes=cfg.metrics.num_classes,
        iou_thrs=cfg.metrics.iou_thrs,
        score_thresh=cfg.metrics.score_thr,
        class_agnostic=cfg.metrics.class_agnostic)

    plotter = TrainingCurvesSemiSupervised(
        metrics_supervised=metric_sup,
        metrics_unsupervised=metric_unsup,
        eval_metrics=eval_metrics)

    best_val_loss = float("inf")

    for epoch in tqdm(range(cfg.train.epochs), desc="Semi-Supervised Epochs"):
        sup_hist, unsup_hist, train_loss = train_semi_supervised_one_epoch(
            teacher=teacher, student=student, optimizer=optimizer, scheduler=lr_scheduler,
            data=data, device=device, max_iter=3000,
            lambda_unsup=cfg.ssl.unsup_weight, score_thr=cfg.ssl.pseudo_conf_thr,
            nms_iou=0.5, metric_sup=metric_sup, metric_unsup=metric_unsup)

        val_hist, val_loss = validate_semi_supervised(
            student=student, dt_test=data["test"], device=device,
            cfg_metrics=cfg_metrics, max_iter=3000)

        early_stopping(value=val_loss, model=student, epoch=epoch + 1)

        plotter.add(
            sup=sup_hist, unsup=unsup_hist, total_loss=train_loss,
            eval_hist=val_hist, val_loss=val_loss)
        plotter.plot_losses(save_dir="graphs", plot_components=True)
        plotter.plot_eval_metrics(save_dir="graphs")

        os.makedirs("unbiased_teacher" + cfg.model.arch + "_checkpoints", exist_ok=True)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_path = os.path.join("unbiased_teacher" + cfg.model.arch + "_checkpoints", f"checkpoint_epoch_{epoch + 1}.pth")
            save_checkpoint(student, optimizer, epoch + 1, ckpt_path)

        if early_stopping.early_stop:
            logger.info("\nEARLY STOPPING TRIGGERED (Semi-Supervised) — restoring best model.\n")
            early_stopping.load_best_model(student)
            ckpt_path = os.path.join("unbiased_teacher" + cfg.model.arch + "_checkpoints", f"checkpoint_epoch_{epoch + 1}.pth")
            save_checkpoint(student, optimizer, epoch + 1, ckpt_path)
            break
