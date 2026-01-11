from __future__ import annotations

from typing import Dict, List
import os
import numpy as np

from torch.utils.data import DataLoader
import torch
from tqdm import tqdm

from utils import save_checkpoint
from bbox import box_iou
from core import (
    move_images_to_device,
    move_targets_to_device,
    mean_history, stats_mean_std)
from utils import (
    visualize_weight_distribution,
    visualize_gradients, 
    visualize_activations,
    plot_dists)
from data.visualize import (
    TrainingCurveSupervised,
    plot_class_distribution, 
    plot_confusion_matrix,
    detect_grid)
from models.builders import build_model, build_scheduler, build_optimizer
from models.hyperparams import ExperimentConfig


def train_burn_in_one_epoch(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    data: Dict[str, DataLoader], device: torch.device,
    max_iter: int, metric_keys: List[str]
) -> Dict[str, float]:
    model.train()
    history = {k: 0.0 for k in metric_keys}
    steps = 0

    loader = data["train_burn_in_strong"]
    for step_idx, (images, targets) in enumerate(tqdm(loader, desc="Burn-in train")):
        if step_idx >= max_iter:
            break

        images = move_images_to_device(images, device)
        targets = move_targets_to_device(targets, device)

        optimizer.zero_grad(set_to_none=True)
        _, loss_dict = model(images, targets)
        if not loss_dict:
            continue

        loss = torch.stack([v for v in loss_dict.values()]).sum()
        loss.backward()
        optimizer.step()
        scheduler.step()

        for k in metric_keys:
            if k != "total" and k in loss_dict:
                history[k] += float(loss_dict[k].item())
        if "total" in history:
            history["total"] += float(loss.item())
        steps += 1

    return mean_history(history, steps)


def collect_labels_from_det_dataset(
    ds: torch.utils.data.Dataset, 
    max_samples: int
) -> List[int]:
    y: List[int] = []
    for i in range(min(len(ds), int(max_samples))):
        item = ds[i]
        if not isinstance(item, (tuple, list)) or len(item) < 2:
            continue
        t = item[1]
        if isinstance(t, dict) and "labels" in t:
            lab = t["labels"]
            if torch.is_tensor(lab):
                y.extend(lab.detach().cpu().tolist())
            elif isinstance(lab, (list, tuple)):
                y.extend([int(v) for v in lab])
    return y


def confusion_from_loader(
    model: torch.nn.Module, loader: DataLoader, device: torch.device,
    num_classes: int, max_batches: int, score_thr: float
) -> np.ndarray:
    model.eval()

    cm = np.zeros((int(num_classes), int(num_classes)), dtype=np.int64)
    seen = 0

    with torch.no_grad():
        for bidx, (images, targets) in enumerate(tqdm(loader, desc="Confusion eval")):
            if bidx >= int(max_batches):
                break

            images = move_images_to_device(images, device)
            preds = model(images) 
            if not isinstance(preds, (list, tuple)):
                continue

            for i in range(len(preds)):
                if i >= len(targets):
                    break

                t, p = targets[i], preds[i]
                gt_labels, gt_boxes = t["labels"], t["boxes"]
                pr_labels, pr_boxes = p["labels"], p["boxes"]
                pr_scores = p.get("scores", None)

                if pr_scores is not None and torch.is_tensor(pr_scores):
                    keep = pr_scores >= float(score_thr)
                    pr_labels = pr_labels[keep]
                    pr_boxes = pr_boxes[keep]

                # It must be a match of no. boxes between pred and gt
                iou = box_iou(pr_boxes.float(), gt_boxes.float())
                best_iou, best_gt = iou.max(dim=1)

                for pi in range(pr_labels.numel()):
                    if float(best_iou[pi].item()) <= 0.0:
                        continue
                    gt_i = int(best_gt[pi].item())
                    pred_c = int(pr_labels[pi].item())
                    gt_c = int(gt_labels[gt_i].item())
                    if 0 <= gt_c < num_classes and 0 <= pred_c < num_classes:
                        cm[gt_c, pred_c] += 1

                seen += 1

    model.train()
    if seen == 0:
        return cm
    return cm


def pipeline_burn_in(
    cfg: ExperimentConfig, data: Dict[str, DataLoader],
    device: torch.device, metric_keys: List[str],
    save_path: str = "images/burn_in"
) -> None:
    model = build_model(cfg=cfg).to(device)
    optimizer = build_optimizer(cfg=cfg, model=model)
    steps_per_epoch = len(data["train_burn_in_strong"])
    lr_scheduler = build_scheduler(
        optimizer=optimizer, scheme=cfg.sched.scheme, 
        total_epochs=cfg.train.epochs, steps_per_epoch=steps_per_epoch,
        warmup_epochs=cfg.sched.warmup_epochs, warmup_bias_lr=cfg.sched.warmup_bias_lr,
        min_lr_ratio=cfg.sched.min_lr_ratio, milestones=cfg.sched.milestones, gamma=cfg.sched.gamma)

    plotter = TrainingCurveSupervised(metrics=metric_keys)

    os.makedirs(save_path, exist_ok=True)

    ckpt_dir = f"model_{cfg.model.arch}_checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)

    ds_train = data["train_burn_in_strong"].dataset
    ds_test = data["test"].dataset

    y_train = collect_labels_from_det_dataset(ds_train, max_samples=500)
    y_val = collect_labels_from_det_dataset(ds_test, max_samples=300)
    class_names = [c.name for c in cfg.data.classes.values()]

    plot_class_distribution(
        y_train=y_train, y_val=y_val, class_names=class_names,
        show=False, save_path=os.path.join(save_path, "class_distribution.png"))

    mean, std = stats_mean_std(ds_train, max_samples=3000)

    for epoch in tqdm(range(cfg.train.epochs), desc="Burn-in epochs"):
        train_hist = train_burn_in_one_epoch(
            model=model, optimizer=optimizer, scheduler=lr_scheduler, data=data, device=device,
            max_iter=(cfg.train.log_interval * 999999), metric_keys=metric_keys)

        plotter.update(train_hist)
        plotter.plot_total(
            save_dir=save_path, show=False,
            save_path=os.path.join(save_path, f"burn_in_{cfg.model.arch}_total.png"))

        if (epoch + 1) % cfg.train.ckpt_interval == 0 or (epoch + 1) == cfg.train.epochs:
            train_set = getattr(data["train_burn_in_strong"], "dataset", None)

            visualize_weight_distribution(
                model, out_dir=save_path,
                file_name=f"burn_in_{cfg.model.arch}_weights_e{epoch + 1}.png")

            if train_set is not None:
                visualize_gradients(
                    model, train_set=train_set, device=device,
                    batch_size=min(256, cfg.data.batch_size), out_dir=save_path,
                    file_name=f"burn_in_{cfg.model.arch}_grads_e{epoch + 1}.png")

                visualize_activations(
                    model, train_set=train_set, device=device,
                    batch_size=min(256, cfg.data.batch_size), out_dir=save_path,
                    file_name=f"burn_in_{cfg.model.arch}_acts_e{epoch + 1}.png")

            plot_dists(
                values={k: np.array([float(train_hist.get(k, 0.0))], dtype=np.float32) for k in train_hist.keys()},
                xlabel="value", out_dir=save_path, file_name=f"burn_in_{cfg.model.arch}_loss_components_e{epoch + 1}.png")

            batch = next(iter(data["train_burn_in_strong"]))
            images, _ = batch
            imgs = torch.stack(images, dim=0) if isinstance(images, list) else images

            if torch.is_tensor(imgs):
                boxes, labels, scores = [], [], []
                with torch.no_grad():
                    preds = model(move_images_to_device(list(imgs), device))

                if isinstance(preds, (list, tuple)):
                    for p in preds:
                        if isinstance(p, dict) and "boxes" in p:
                            boxes.append(p["boxes"].detach().cpu())
                            labels.append(p["labels"].detach().cpu())
                            scores.append(p["scores"].detach().cpu())

                    detect_grid(
                        images=imgs.detach().cpu(), boxes=boxes, labels=labels, scores=scores, classes=getattr(cfg.data, "classes", None),
                        mean=mean, std=std, pred_status=None, titles=None, conf_thr=cfg.metrics.score_thr, grid_title=f"Burn-in preds e{epoch + 1}",
                        cols=4, figsize_per_cell=(3.3, 3.3), show=False, save_path=os.path.join(save_path, f"burn_in_{cfg.model.arch}_grid_e{epoch + 1}.png"))

            cm = confusion_from_loader(
                model=model, loader=data["test"], device=device,
                num_classes=int(cfg.data.num_classes), max_batches=5, score_thr=cfg.metrics.score_thr)

            plot_confusion_matrix(
                cm=cm, class_names=class_names, title=f"Burn-in confusion e{epoch + 1}",
                normalize=True, show=False, save_path=os.path.join(save_path, f"burn_in_{cfg.model.arch}_cm_e{epoch + 1}.png"))

            ckpt_path = os.path.join(ckpt_dir, f"checkpoint_epoch_{epoch + 1}.pth")
            save_checkpoint(model=model, optimizer=optimizer, epoch=epoch + 1, path=ckpt_path)
