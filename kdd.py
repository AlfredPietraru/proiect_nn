from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from core import mean_history
from utils.checkpoints import load_checkpoint, save_checkpoint
from data.visualize.training_curves import TrainingCurveSupervised
from models.scheduler import build_scheduler
from models.hyperparams import ExperimentConfig
from model_builder import build_model
from models.kl import WeakStrongKDD, CrossDatasetKDD, ClassProjector, FeatureKDD, BoxMatchKDD


def _pick(model: nn.Module, *names: str):
    for n in names:
        if hasattr(model, n) and callable(getattr(model, n)):
            return getattr(model, n)
    return None


def _init_kdd(cfg: ExperimentConfig, device: torch.device):
    kind = cfg.kdd.kind
    w_cls, w_feat, w_box = float(cfg.kdd.w_cls), float(cfg.kdd.w_feat), float(cfg.kdd.w_box)

    # default: single-mode uses weight 1 on that part if user left others 0
    if kind != "combo":
        w_cls, w_feat, w_box = (1.0, 0.0, 0.0)
        if kind == "feature":
            w_cls, w_feat, w_box = (0.0, 1.0, 0.0)
        if kind == "box_match":
            w_cls, w_feat, w_box = (0.0, 0.0, 1.0)

    kdd_cls: Optional[nn.Module] = None
    if kind in ("weakstrong", "combo"):
        kdd_cls = WeakStrongKDD(tau=cfg.kdd.tau, gamma=cfg.kdd.gamma, eps=cfg.kdd.eps).to(device)

    if kind == "cross_dataset":
        if cfg.kdd.teacher_to_student is None:
            raise RuntimeError("cfg.kdd.teacher_to_student is required for kind='cross_dataset'.")
        proj = ClassProjector(teacher_to_student=cfg.kdd.teacher_to_student, ks=int(cfg.data.num_classes)).to(device)
        kdd_cls = CrossDatasetKDD(projector=proj, tau=cfg.kdd.tau, gamma=cfg.kdd.gamma, eps=cfg.kdd.eps).to(device)

    kdd_feat: Optional[nn.Module] = None
    if kind in ("feature", "combo"):
        kdd_feat = FeatureKDD(proj=nn.Identity(), beta=cfg.kdd.beta).to(device)

    kdd_box: Optional[nn.Module] = None
    if kind in ("box_match", "combo"):
        kdd_box = BoxMatchKDD(
            tau=cfg.kdd.tau, gamma=cfg.kdd.gamma,
            iou_thr=cfg.kdd.iou_thr, box_l1=cfg.kdd.box_l1
        ).to(device)

    return kind, (w_cls, w_feat, w_box), kdd_cls, kdd_feat, kdd_box


def train_kdd_one_epoch(
    teacher: nn.Module,
    student: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    data: Dict[str, DataLoader],
    device: torch.device,
    max_iter: int,
    metric_keys: List[str], *,
    kind: str,
    weights: Tuple[float, float, float],
    kdd_cls: Optional[nn.Module],
    kdd_feat: Optional[nn.Module],
    kdd_box: Optional[nn.Module],
) -> Dict[str, float]:
    teacher.eval()
    student.train()

    history = {k: 0.0 for k in metric_keys}
    steps = 0

    w_cls, w_feat, w_box = weights
    loader = data["train_weak"]

    # hooks are optional; we only require them if their loss is enabled
    get_feat_t = _pick(teacher, "extract_features", "forward_features")
    get_feat_s = _pick(student, "extract_features", "forward_features")
    get_box_t = _pick(teacher, "predict_boxes_logits")
    get_box_s = _pick(student, "predict_boxes_logits")

    for step_idx, (img_weak, img_strong) in enumerate(tqdm(loader, desc=f"KDD ({kind}) train")):
        if step_idx >= max_iter:
            break

        if not torch.is_tensor(img_weak) or not torch.is_tensor(img_strong):
            raise RuntimeError("KDD expects tensor batches (img_weak, img_strong) as NCHW tensors.")

        xw = img_weak.to(device, non_blocking=True)
        xs = img_strong.to(device, non_blocking=True)

        loss_total = xs.new_zeros(())

        # --- logits KL ---
        if kdd_cls is not None and w_cls > 0.0:
            with torch.no_grad():
                t_logits = teacher(xw)
            s_logits = student(xs)
            loss_kl, conf, w = kdd_cls(teacher_logits_w=t_logits, student_logits_s=s_logits, weight=None)
            loss_total = loss_total + w_cls * loss_kl

            if "kdd_kl" in history:
                history["kdd_kl"] += float(loss_kl.item())
            if "kdd_conf" in history:
                history["kdd_conf"] += float(conf.mean().item())
            if "kdd_w" in history:
                history["kdd_w"] += float(w.mean().item())

        # --- feature distill ---
        if kdd_feat is not None and w_feat > 0.0:
            if get_feat_t is None or get_feat_s is None:
                raise RuntimeError("FeatureKDD requires extract_features(x) or forward_features(x) on both teacher and student.")
            with torch.no_grad():
                f_t = get_feat_t(xw)
            f_s = get_feat_s(xs)
            loss_f = kdd_feat(f_t=f_t, f_s=f_s)
            loss_total = loss_total + w_feat * loss_f
            if "kdd_feat" in history:
                history["kdd_feat"] += float(loss_f.item())

        # --- box match distill ---
        if kdd_box is not None and w_box > 0.0:
            if get_box_t is None or get_box_s is None:
                raise RuntimeError("BoxMatchKDD requires predict_boxes_logits(x)->(boxes, logits, valid) on teacher and student.")
            with torch.no_grad():
                t_boxes, t_logits, t_valid = get_box_t(xw)
            s_boxes, s_logits, s_valid = get_box_s(xs)
            loss_b = kdd_box(
                t_boxes=t_boxes, t_logits=t_logits, t_valid=t_valid,
                s_boxes=s_boxes, s_logits=s_logits, s_valid=s_valid
            )
            loss_total = loss_total + w_box * loss_b
            if "kdd_box" in history:
                history["kdd_box"] += float(loss_b.item())

        optimizer.zero_grad(set_to_none=True)
        loss_total.backward()
        optimizer.step()
        scheduler.step()

        history["total"] += float(loss_total.item())
        steps += 1

    return mean_history(history, steps)


def pipeline_kdd(
    cfg: ExperimentConfig,
    data: Dict[str, DataLoader],
    device: torch.device,
    teacher_ckpt: str,  # path to teacher checkpoint - from unbiased teacher
    student_ckpt: str,  # path to student checkpoint - from unbiased teacher
    metric_keys: List[str],
) -> None:
    teacher = build_model(cfg=cfg).to(device)
    student = build_model(cfg=cfg).to(device)

    teacher, _, _ = load_checkpoint(path=teacher_ckpt, model=teacher, optimizer=None, device=device)
    student, _, _ = load_checkpoint(path=student_ckpt, model=student, optimizer=None, device=device)

    for p in teacher.parameters():
        p.requires_grad_(False)

    optimizer = torch.optim.SGD(
        student.parameters(), lr=cfg.optim.lr, momentum=cfg.optim.momentum,
        weight_decay=cfg.optim.weight_decay, nesterov=cfg.optim.nesterov)

    steps_per_epoch = len(data["train_weak"])
    lr_scheduler = build_scheduler(
        optimizer=optimizer,
        scheme=cfg.sched.scheme, total_epochs=cfg.train.epochs, steps_per_epoch=steps_per_epoch,
        warmup_epochs=cfg.sched.warmup_epochs, warmup_bias_lr=cfg.sched.warmup_bias_lr,
        min_lr_ratio=cfg.sched.min_lr_ratio, milestones=cfg.sched.milestones, gamma=cfg.sched.gamma)

    kind, weights, kdd_cls, kdd_feat, kdd_box = _init_kdd(cfg, device)
    plotter = TrainingCurveSupervised(metrics=metric_keys)

    for epoch in tqdm(range(cfg.train.epochs), desc=f"KDD epochs ({kind})"):
        train_hist = train_kdd_one_epoch(
            teacher=teacher, student=student,
            optimizer=optimizer, scheduler=lr_scheduler, data=data, device=device,
            max_iter=(cfg.train.log_interval * 999999), metric_keys=metric_keys,
            kind=kind, weights=weights, kdd_cls=kdd_cls, kdd_feat=kdd_feat, kdd_box=kdd_box)

        plotter.plot_total(epoch_history=train_hist, save_dir="graphs")

        if (epoch + 1) % cfg.train.ckpt_interval == 0 or (epoch + 1) == cfg.train.epochs:
            os.makedirs("kdd" + cfg.model.arch + "_checkpoints", exist_ok=True)
            ckpt_path = os.path.join("kdd" + cfg.model.arch + "_checkpoints", f"checkpoint_epoch_{epoch + 1}.pth")
            save_checkpoint(student, optimizer, epoch + 1, ckpt_path)
