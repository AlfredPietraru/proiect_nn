from __future__ import annotations

from typing import Dict, List
import os
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm

from utils.checkpoints import save_checkpoint
from core import move_images_to_device, move_targets_to_device, mean_history
from models.scheduler import build_scheduler
from models.hyperparams import ExperimentConfig
from model_builder import build_model
from data.visualize.training_curves import TrainingCurveSupervised


def train_burn_in_one_epoch(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    data: Dict[str, DataLoader],
    device: torch.device,
    max_iter: int,
    metric_keys: List[str],
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

        loss = sum(loss_dict.values())
        loss.backward()
        optimizer.step()
        scheduler.step()

        for k in metric_keys:
            if k != "total" and k in loss_dict:
                history[k] += float(loss_dict[k].item())
        history["total"] += float(loss.item())
        steps += 1

    return mean_history(history, steps)


def pipeline_burn_in(
    cfg: ExperimentConfig,
    data: Dict[str, DataLoader],
    device: torch.device,
    metric_keys: List[str],
) -> None:
    model = build_model(cfg=cfg).to(device) 

    optimizer = torch.optim.SGD(
        model.parameters(), lr=cfg.optim.lr, momentum=cfg.optim.momentum,
        weight_decay=cfg.optim.weight_decay, nesterov=cfg.optim.nesterov)

    steps_per_epoch = len(data["train_burn_in_strong"])
    lr_scheduler = build_scheduler(
        optimizer=optimizer,
        scheme=cfg.sched.scheme, total_epochs=cfg.train.epochs, steps_per_epoch=steps_per_epoch,
        warmup_epochs=cfg.sched.warmup_epochs, warmup_bias_lr=cfg.sched.warmup_bias_lr,
        min_lr_ratio=cfg.sched.min_lr_ratio, milestones=cfg.sched.milestones, gamma=cfg.sched.gamma)

    plotter = TrainingCurveSupervised(metrics=metric_keys)

    for epoch in tqdm(range(cfg.train.epochs), desc="Burn-in epochs"):
        train_hist = train_burn_in_one_epoch(
            model=model, optimizer=optimizer, scheduler=lr_scheduler, data=data, device=device,
            max_iter=(cfg.train.log_interval * 999999), metric_keys=metric_keys)
        plotter.update(epoch_metrics=train_hist)
        plotter.plot_total(save_dir="graphs")

        if (epoch + 1) % cfg.train.ckpt_interval == 0 or (epoch + 1) == cfg.train.epochs:
            os.makedirs("burn_in" + cfg.model.arch + "_checkpoints", exist_ok=True)
            ckpt_path = os.path.join("burn_in" + cfg.model.arch + "_checkpoints", f"checkpoint_epoch_{epoch + 1}.pth")
            save_checkpoint(model, optimizer, epoch + 1, ckpt_path)
