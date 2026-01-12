from __future__ import annotations

import os
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.ops import batched_nms
from tqdm import tqdm
from loguru import logger

from utils import (
    visualize_weight_distribution,
    visualize_gradients, visualize_activations,
    plot_dists)
from utils import load_checkpoint, save_checkpoint
from core import (
    move_images_to_device, move_targets_to_device, 
    stats_mean_std)
from models.builders import build_model, build_scheduler, build_optimizer
from bbox import BoxList, box_iou
from models.early_stopping import EarlyStopping
from models.hyperparams import ExperimentConfig
from models.ema import EMA
from evaluate import DetectionMetrics, IoUMetrics
from data.visualize import (
    TrainingCurveSemiSupervised,
    plot_agreement_teacher_vs_student,
    plot_confusion_matrix, detect_grid)
from models.gradcam_eval import evaluate_cam_bboxes


def top_k_per_class(
    boxes: torch.Tensor,
    labels: torch.Tensor,
    scores: torch.Tensor,
    k: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Keep top-k scoring boxes per class."""
    if k is None:
        return boxes, labels, scores
    # Keep top-k scoring boxes per class
    keep_idx: List[torch.Tensor] = []
    for cls in labels.unique():
        # Get indices of boxes for this class and their scores
        cls_idx = (labels == cls).nonzero(as_tuple=False).squeeze(1)
        cls_scores = scores[cls_idx]
        # Keep top-k for this class based on scores
        if cls_scores.numel() <= k:
            keep_idx.append(cls_idx)
        else:
            top_idx = cls_idx[torch.topk(cls_scores, k, sorted=False).indices]
            keep_idx.append(top_idx)
    # Concatenate all kept indices
    if not keep_idx:
        return boxes[:0], labels[:0], scores[:0]
    keep = torch.cat(keep_idx, dim=0)
    return boxes[keep], labels[keep], scores[keep]


def top_k_total(
    boxes: torch.Tensor,
    labels: torch.Tensor,
    scores: torch.Tensor,
    k: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Keep top-k scoring boxes total (regardless of class)."""
    if k is None or scores.numel() <= k:
        return boxes, labels, scores
    idx = torch.topk(scores, k, sorted=False).indices
    return boxes[idx], labels[idx], scores[idx]


@torch.no_grad()
def generate_pseudo_labels(
    model: torch.nn.Module,
    images: List[torch.Tensor],
    device: torch.device,
    score_thr: float, nms_iou: float,
    top_k_per_cls: int = 60, # per-class candidates before NMS
    top_k_total_pre: int = 600, # total candidates per image before NMS
    top_k_total_post: int = 300 # final max per image after NMS
) -> List[Dict[str, torch.Tensor]]:
    model.eval()
    images = move_images_to_device(images, device)
    outputs, _ = model(images, None)
    pseudo: List[Dict[str, torch.Tensor]] = []
    for out in outputs:
        boxes, labels, scores = out["boxes"], out["labels"], out["scores"]
        # Early exit if no boxes at all
        if boxes.numel() == 0:
            pseudo.append({"boxes": boxes, "labels": labels, "scores": scores})
            continue
        # Score thresholding - remove the most low confidence preds
        keep = scores > score_thr
        boxes, labels, scores = boxes[keep], labels[keep], scores[keep]
        if boxes.numel() == 0:
            pseudo.append({"boxes": boxes, "labels": labels, "scores": scores})
            continue
        # Top-k total per image (pre NMS)
        boxes, labels, scores = top_k_total(boxes, labels, scores, top_k_total_pre)
        # Top-k per class (pre NMS) 
        boxes, labels, scores = top_k_per_class(boxes, labels, scores, top_k_per_cls)
        # Class-aware NMS
        keep = batched_nms(boxes, scores, labels, iou_threshold=nms_iou)
        boxes, labels, scores = boxes[keep], labels[keep], scores[keep]
        # Score threshold again (safe)
        keep = scores > score_thr
        boxes, labels, scores = boxes[keep], labels[keep], scores[keep]
        # Top-k total per image (post NMS, FINAL CAP)
        boxes, labels, scores = top_k_total(boxes, labels, scores, top_k_total_post)
        pseudo.append({"boxes": boxes.detach(), "labels": labels.detach(), "scores": scores.detach()})
    return pseudo



def filter_nonempty_pseudo(
    pseudo: List[Dict[str, torch.Tensor]]
) -> Tuple[List[Dict[str, torch.Tensor]], List[int]]:
    """Filter out images with no pseudo boxes, return kept pseudo and their indices."""
    keep_idx = [i for i, t in enumerate(pseudo) if t["boxes"].numel() > 0]
    kept = [pseudo[i] for i in keep_idx]  # Keep only pseudo-labels with non-empty boxes
    kept = [{"boxes": t["boxes"], "labels": t["labels"]} for t in kept]
    return kept, keep_idx


def train_semi_supervised_one_epoch(
    teacher: EMA, student: torch.nn.Module,  # teacher is wrapped with EMA
    optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler._LRScheduler,
    data: Dict[str, DataLoader], device: torch.device,  max_iter: int,
    lambda_unsup: float, score_thr: float, nms_iou: float,
    metric_sup: List[str], metric_unsup: List[str]
) -> Tuple[Dict[str, float], Dict[str, float], float]:
    student.train()

    hist_sup = {k: 0.0 for k in metric_sup}
    hist_unsup = {k: 0.0 for k in metric_unsup}
    total_loss = 0.0
    steps = 0

    unlabeled_loader = data["train_weak"]
    labeled_loader = data["train_burn_in_strong"]
    labeled_it = iter(labeled_loader)

    for step_idx, (img_weak, img_strong) in enumerate(tqdm(unlabeled_loader, desc="SSL train")):

        # Generate pseudo-labels with teacher model on weakly augmented images
        pseudo = generate_pseudo_labels(
            model=teacher.ema, images=img_weak, device=device,
            score_thr=score_thr, nms_iou=nms_iou)

        # Filter out images with no pseudo-labels to avoid useless computation
        pseudo_kept, keep_idx = filter_nonempty_pseudo(pseudo)
        if not keep_idx:
            continue

        # Move strong images and kept pseudo-labels to device
        img_strong = move_images_to_device(img_strong, device)
        img_strong = [img_strong[i] for i in keep_idx]
        pseudo_kept = move_targets_to_device(pseudo_kept, device)

        # Student forward pass on strongly augmented images with pseudo-labels
        _, loss_u = student(img_strong, pseudo_kept)
        unsup_cls = loss_u["loss_classifier"] + loss_u["loss_objectness"]

        # Supervised loss on labeled data (batch from strong loader)
        img_labeled, targets_labeled = next(labeled_it)
        img_labeled = move_images_to_device(img_labeled, device)
        targets_labeled = move_targets_to_device(targets_labeled, device)

        # Student forward pass on labeled data
        _, loss_s = student(img_labeled, targets_labeled)
        sup = sum(loss_s.values())
        loss = sup + lambda_unsup * unsup_cls

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()

        # Update teacher EMA model with current student weights
        teacher.update(student)

        # Accumulate metrics for logging after epoch
        for k in metric_unsup:
            if k in loss_u:
                hist_unsup[k] += float(loss_u[k].item())
        for k in metric_sup:
            if k in loss_s:
                hist_sup[k] += float(loss_s[k].item())

        total_loss += float(loss.item())
        steps += 1

    # Average the metrics over steps
    denom = max(1, steps)
    for k in hist_unsup:
        hist_unsup[k] /= denom
    for k in hist_sup:
        hist_sup[k] /= denom

    return hist_sup, hist_unsup, total_loss / denom


# @torch.no_grad()
def validate_semi_supervised(
    student: torch.nn.Module, dt_test: DataLoader,
    device: torch.device, cfg_metrics: IoUMetrics,
    max_iter: int = 3000, nms_iou: float = 0.5
) -> Tuple[Dict[str, float], float]:
    student.eval()

    metrics = DetectionMetrics(cfg_metrics)
    metrics.reset()

    loss_sum = 0.0

    for step_idx, (images, targets) in enumerate(tqdm(dt_test, desc="Validation")):

        images = move_images_to_device(images, device)
        targets = move_targets_to_device(targets, device)
        # Get student outputs + loss dict
        outputs, loss_dict = student(images, targets)
        if loss_dict:
            loss_sum += sum(v.item() for v in loss_dict.values())

        preds_bl: List[BoxList] = []
        tgts_bl: List[BoxList] = []
        print()
        print(loss_dict)
        print("Lungimile rezultatelor: ", len(outputs), " ", len(targets), end="\n\n")
        print()

        # Checking for each image in the batch the predictions and targets
        for img, out, tgt in zip(images, outputs, targets):
            h, w = int(img.shape[-2]), int(img.shape[-1])
            size_hw = (h, w)
            # Predictions (pseudo)
            out_boxes = out.get("boxes", torch.zeros((0, 4), device))
            out_labels = out.get("labels", torch.zeros((0,), torch.int64, device))
            out_scores = out.get("scores", None)
            # Targets (ground-truth)
            tgt_boxes = tgt.get("boxes", torch.zeros((0, 4), device))
            tgt_labels = tgt.get("labels", torch.zeros((0,), torch.int64, device))
            tgt_scores = tgt.get("scores", None)
            # NMS on predictions before evaluation
            if out_boxes.numel() > 0 and out_scores is not None:
                keep = batched_nms(out_boxes, out_scores, out_labels, iou_threshold=nms_iou)
                out_boxes, out_labels, out_scores = out_boxes[keep], out_labels[keep], out_scores[keep]
            preds_bl.append(BoxList(out_boxes, out_labels, out_scores, size_hw))
            tgts_bl.append(BoxList(tgt_boxes, tgt_labels, tgt_scores, size_hw))

        # Update metrics with batch predictions and targets
        metrics.update(preds_bl, tgts_bl)

    avg_loss = loss_sum / max(1, len(dt_test))
    return metrics.compute(), avg_loss


def pipeline_semi_supervised(
    cfg: ExperimentConfig, checkpoint_path: str,
    data: Dict[str, DataLoader], device: torch.device,
    metric_sup: List[str], metric_unsup: List[str]
) -> None:
    # Student - initialize model from checkpoint from burn-in stage
    # Load burn-in checkpoint into student model
    student = build_model(cfg=cfg).to(device)
    student, _, _ = load_checkpoint(checkpoint_path, student, optimizer=None, device=device)
    # Teacher - Same model architecture for now wrapped in EMA (Exponential Moving Average)
    teacher = EMA(student, decay=cfg.ssl.ema_decay)

    # Early Stopping based on validation loss for student model learning
    early_stopping = EarlyStopping(patience=8, min_delta=1e-3, mode="min", verbose=True)

    # Optimizer + LR Scheduler for student
    # Dependent of no. steps per epoch (based on weak data loader)
    steps_per_epoch = len(data["train_weak"])
    optimizer = build_optimizer(cfg=cfg, model=student)
    scheduler = build_scheduler(
        optimizer=optimizer, scheme=cfg.sched.scheme, 
        total_epochs=cfg.train.epochs, steps_per_epoch=steps_per_epoch,
        warmup_epochs=cfg.sched.warmup_epochs, warmup_bias_lr=cfg.sched.warmup_bias_lr,
        min_lr_ratio=cfg.sched.min_lr_ratio, milestones=cfg.sched.milestones, gamma=cfg.sched.gamma)

    # Metrics for evaluation of student model during SSL training
    # - IoU-based metrics for object detection (mAP50, mAP50_95, etc.)
    cfg_metrics = IoUMetrics(
        num_classes=cfg.metrics.num_classes,
        iou_thrs=cfg.metrics.iou_thrs,
        score_thresh=cfg.metrics.score_thr,
        class_agnostic=cfg.metrics.class_agnostic)

    # Results logging + visualization setup
    graphs_dir = "graphs"
    os.makedirs(graphs_dir, exist_ok=True)
    ckpt_dir = f"model_{cfg.model.arch}_checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)

    # Semi-supervised training loop (student-teacher with EMA)
    plotter = TrainingCurveSemiSupervised(metric_sup, metric_unsup)

    ds_train_weak = data["train_weak"].dataset
    # Obtain specific details of the images used in the current dataset
    mean, std = stats_mean_std(ds_train_weak, max_samples=3000)
    eval_history = {k: [] for k in cfg_metrics.metric_names()}

    best_val_loss = float("inf")
    for epoch in tqdm(range(cfg.train.epochs), desc="Semi-Supervised Epochs"):
        # Training step for one epoch of semi-supervised learning
        sup_hist, unsup_hist, train_loss = train_semi_supervised_one_epoch(
            scheduler=scheduler, optimizer=optimizer,
            teacher=teacher, student=student,
            lambda_unsup=cfg.ssl.unsup_weight, 
            data=data, device=device, score_thr=cfg.ssl.pseudo_conf_thr,
            metric_sup=metric_sup, metric_unsup=metric_unsup, max_iter=3000, nms_iou=0.5)

        # Checking student performance on validation set after this epoch
        val_hist, val_loss = validate_semi_supervised(
            student=student, dt_test=data["test"], device=device,
            cfg_metrics=cfg_metrics, max_iter=3000, nms_iou=0.5)

        early_stopping(value=val_loss, model=student, epoch=epoch + 1)
        # Plottng current training/validation metrics after this epoch
        plotter.update_supervised(sup_hist)
        plotter.update_unsupervised(unsup_hist)
        plotter.update_total_loss(train_loss)

        # Logging evaluation metrics
        for k in eval_history:
            eval_history[k].append(float(val_hist.get(k, 0.0)))

        plotter.plot_losses(
            plot_components=True, save_dir=graphs_dir, show=False,
            save_path=os.path.join(graphs_dir, f"ssl_{cfg.model.arch}_losses.png"))
        plotter.plot_eval_metrics(
            metrics=eval_history, save_dir=graphs_dir, show=False,
            save_path=os.path.join(graphs_dir, f"ssl_{cfg.model.arch}_eval.png"))

        if (epoch + 1) % cfg.train.ckpt_interval == 0 or (epoch + 1) == cfg.train.epochs:
            train_set = getattr(data["train_burn_in_strong"], "dataset", None)

            # Visualizations for analysis of training dynamics and model behavior
            visualize_weight_distribution(
                student, out_dir=graphs_dir,
                file_name=f"ssl_{cfg.model.arch}_weights_e{epoch + 1}.png")

            # Gradients and activations visualization (on labeled data)
            if train_set is not None:
                visualize_gradients(
                    student, train_set=train_set, device=device,
                    batch_size=min(256, cfg.data.batch_size), out_dir=graphs_dir,
                    file_name=f"ssl_{cfg.model.arch}_grads_e{epoch + 1}.png")
                visualize_activations(
                    student, train_set=train_set, device=device,
                    batch_size=min(256, cfg.data.batch_size), out_dir=graphs_dir,
                    file_name=f"ssl_{cfg.model.arch}_acts_e{epoch + 1}.png")

            # Distribution plots for pseudo-label statistics (weak data)
            img_weak, _ = next(iter(data["train_weak"]))
            pseudo = generate_pseudo_labels(
                model=teacher.ema, images=img_weak, device=device,
                score_thr=cfg.ssl.pseudo_conf_thr, nms_iou=0.5)

            # Collect scores and box counts for distribution plots
            all_scores, per_img_counts = [], []
            for p in pseudo:
                s = p.get("scores", None)
                b = p.get("boxes", None)
                if s is not None and torch.is_tensor(s):
                    all_scores.extend(s.detach().cpu().tolist())
                if b is not None and torch.is_tensor(b):
                    per_img_counts.append(int(b.size(0)))

            # Plot distributions of pseudo-label scores and boxes per image
            vals = {}
            if len(all_scores) > 0:
                vals["pseudo_scores"] = np.asarray(all_scores, np.float32)
            if len(per_img_counts) > 0:
                vals["pseudo_boxes_per_img"] = np.asarray(per_img_counts, np.float32)

            # Only plot if we have some values to show of the pseudo-labels
            # It checks the quality of the pseudo-labels generated by the teacher model
            if len(vals) > 0:
                plot_dists(
                    values=vals, xlabel="value", out_dir=graphs_dir,
                    file_name=f"ssl_{cfg.model.arch}_pseudo_dists_e{epoch + 1}.png")

            # Logits are calculated on weak images for both teacher and student
            # Checking the agreement between teacher and student predictions
            with torch.no_grad():
                out_t, _ = teacher.ema(move_images_to_device(img_weak, device), None)
                out_s, _ = student(move_images_to_device(img_weak, device), None)

            teacher_logits, student_logits = None, None
            if isinstance(out_t, list) and len(out_t) > 0 and isinstance(out_t[0], dict) and "logits" in out_t[0]:
                teacher_logits = out_t[0]["logits"]
            if isinstance(out_s, list) and len(out_s) > 0 and isinstance(out_s[0], dict) and "logits" in out_s[0]:
                student_logits = out_s[0]["logits"]

            # Plot teacher vs student agreement if logits are available for both
            if teacher_logits is not None and student_logits is not None:
                plot_agreement_teacher_vs_student(
                    teacher_logits=teacher_logits, student_logits=student_logits,
                    class_names=[c.name for c in cfg.data.classes.values()],
                    show=False, title="Teacher vs Student Agreement",
                    save_path=os.path.join(graphs_dir, f"ema_teacher_student_agreement_e{epoch + 1}.png"))

            # Qualitative grid (teacher pseudo predictions on weak batch)
            imgs = torch.stack(img_weak, dim=0) if isinstance(img_weak, list) else img_weak
            if torch.is_tensor(imgs):
                boxes = [p["boxes"].detach().cpu() for p in pseudo]
                labels = [p["labels"].detach().cpu() for p in pseudo]
                scores = [p.get("scores", None) if p.get("scores", None) is not None else None for p in pseudo]
                detect_grid(
                    images=imgs.detach().cpu(),
                    boxes=boxes, labels=labels, scores=scores,
                    classes=cfg.data.classes, cols=4, figsize_per_cell=(3.3, 3.3), 
                    mean=mean, std=std, pred_status=None, titles="Teacher Pseudo Labels",
                    conf_thr=float(cfg.ssl.pseudo_conf_thr), grid_title=f"SSL pseudo (teacher EMA) e{epoch + 1}",
                    show=False, save_path=os.path.join(graphs_dir, f"ssl_{cfg.model.arch}_pseudo_grid_e{epoch + 1}.png"))

            # Confusion matrix is SSL-stage relevant (evaluate student on test).
            cm = np.zeros((int(cfg.metrics.num_classes), int(cfg.metrics.num_classes)), dtype=np.int64)

            with torch.no_grad():
                for bidx, (images, targets) in enumerate(data["test"]):
                    if bidx >= (cfg.data.batch_size / 4):  # limit no. batches for speed in evaluation
                        break

                    # Checking results of student model from SSL training
                    images = move_images_to_device(images, device)
                    targets = move_targets_to_device(targets, device)
                    outputs, _ = student(images, None)

                    for out, tgt, _ in zip(outputs, targets, images):
                        pr_scores = out.get("scores", None)
                        # Predicted boxes and labels
                        pr_boxes, pr_labels = out["boxes"], out["labels"]
                        # Ground-truth boxes and labels
                        gt_boxes, gt_labels = tgt["boxes"], tgt["labels"]

                        # Score thresholding on predictions for confusion matrix
                        if pr_scores is not None and torch.is_tensor(pr_scores):
                            keep = pr_scores >= float(cfg.metrics.score_thr)
                            pr_boxes, pr_labels = pr_boxes[keep], pr_labels[keep]
                        if pr_boxes.numel() == 0 or gt_boxes.numel() == 0:
                            continue

                        # Match predictions to ground-truths based on IoU
                        iou = box_iou(pr_boxes.float(), gt_boxes.float())
                        best_iou, best_gt = iou.max(dim=1)

                        # Accumulate confusion matrix counts based on matches
                        # Building over time based on samples from the smaller
                        # validation dataset batches
                        for i in range(pr_labels.numel()):
                            if float(best_iou[i].item()) <= 0.0:
                                continue
                            gi = int(best_gt[i].item())
                            pc = int(pr_labels[i].item())
                            gc = int(gt_labels[gi].item())
                            if 0 <= gc < cm.shape[0] and 0 <= pc < cm.shape[1]:
                                cm[gc, pc] += 1

            # For Grad-CAM models, perform bbox evaluation at each checkpoint
            if cfg.model.arch == "resnet_gradcam":
                metrics: Dict[str, float] = {}
                if hasattr(data["val"], "dataset"):
                    val_set = data["val"].dataset
                    metrics = evaluate_cam_bboxes(
                        campp=student.campp, model=student.backbone,
                        images=val_set.images.to(device, non_blocking=True),
                        gt_boxes=val_set.gt_boxes.to(device, non_blocking=True),
                        gt_labels=val_set.gt_labels.to(device, non_blocking=True),
                        iou_thr=0.5, cam_thr=0.35)
                print(f"KDD epoch {epoch + 1} Grad-CAM evaluation metrics: {metrics}")

            plot_confusion_matrix(
                cm=cm, class_names=[c.name for c in cfg.data.classes.values()],
                title=f"SSL confusion e{epoch + 1}", normalize=True, show=False,
                save_path=os.path.join(graphs_dir, f"ssl_{cfg.model.arch}_cm_e{epoch + 1}.png"))

        # Improvement check for best model saving based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_path = os.path.join(ckpt_dir, f"checkpoint_epoch_{epoch + 1}.pth")
            save_checkpoint(model=student, optimizer=optimizer, epoch=epoch + 1, path=ckpt_path)

        # Early stopping check for semi-supervised training
        if early_stopping.early_stop:
            logger.info("\nEARLY STOPPING TRIGGERED (Semi-Supervised) — restoring best model.\n")
            early_stopping.load_best_model(student)
            ckpt_path = os.path.join(ckpt_dir, f"checkpoint_epoch_{epoch + 1}.pth")
            save_checkpoint(model=student, optimizer=optimizer, epoch=epoch + 1, path=ckpt_path)
            break
