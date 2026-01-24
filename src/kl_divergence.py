from __future__ import annotations

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from models import (
    ExperimentConfig,
    WeakStrongKDD, CrossDatasetKDD,
    ClassProjector, FeatureKDD, BoxMatchKDD,
    evaluate_cam_bboxes,
    build_model, build_optimizer, build_scheduler)
from core import mean_history
from utils import (
    load_checkpoint, save_checkpoint,
    visualize_weight_distribution, visualize_gradients, visualize_activations)
from data import (
    TrainingCurveSupervised,
    kl_divergence, plot_kl_stagewise,
    plot_cross_arch_kl, plot_cross_dataset_kl,
    agreement_matrix, plot_agreement_heatmap, plot_agreement_ema_vs_kdd)


def softmax_mean(logits: torch.Tensor) -> np.ndarray:
    """Useful for avg prediction distribution over a batch."""
    p = torch.softmax(logits.float(), dim=1)
    p = p.mean(dim=0)
    return p.detach().cpu().numpy().astype(np.float64)


def init_kdd(cfg: ExperimentConfig, device: torch.device):
    kind = cfg.kdd.kind
    w_cls, w_feat, w_box = float(cfg.kdd.w_cls), float(cfg.kdd.w_feat), float(cfg.kdd.w_box)

    if kind != "combo":
        w_cls, w_feat, w_box = (1.0, 0.0, 0.0)
        if kind == "feature":
            w_cls, w_feat, w_box = (0.0, 1.0, 0.0)
        if kind == "box_match":
            w_cls, w_feat, w_box = (0.0, 0.0, 1.0)

    kdd_cls: nn.Module | None = None
    if kind in ("weakstrong", "combo"):
        kdd_cls = WeakStrongKDD(tau=cfg.kdd.tau, gamma=cfg.kdd.gamma).to(device)

    if kind == "cross_dataset":
        if cfg.kdd.teacher_to_student is None:
            raise ValueError("Cross-dataset requires mapping between teacher and student classes.")
        proj = ClassProjector(teacher_to_student=cfg.kdd.teacher_to_student, ks=int(cfg.data.num_classes)).to(device)
        kdd_cls = CrossDatasetKDD(projector=proj, tau=cfg.kdd.tau, gamma=cfg.kdd.gamma).to(device)

    kdd_feat: nn.Module | None = None
    if kind in ("feature", "combo"):
        kdd_feat = FeatureKDD(proj=nn.Identity(), beta=cfg.kdd.beta).to(device)

    kdd_box: nn.Module | None = None
    if kind in ("box_match", "combo"):
        kdd_box = BoxMatchKDD(
            tau=cfg.kdd.tau, gamma=cfg.kdd.gamma, 
            iou_thr=cfg.kdd.iou_thr, box_l1=cfg.kdd.box_l1).to(device)

    return kind, (w_cls, w_feat, w_box), kdd_cls, kdd_feat, kdd_box


def train_kdd_one_epoch(
    teacher: nn.Module, student: nn.Module, 
    optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler._LRScheduler, 
    data: dict[str, DataLoader], device: torch.device, max_iter: int,
    metric_keys: list[str], kind: str, weights: tuple[float, float, float],
    kdd_cls: nn.Module | None, kdd_feat: nn.Module | None, kdd_box: nn.Module | None
) -> dict[str, float]:
    teacher.eval()
    student.train()

    history = {k: 0.0 for k in metric_keys}
    steps = 0

    w_cls, w_feat, w_box = weights
    loader = data["train_weak"]

    get_feat_t = teacher.extract_features
    get_feat_s = student.extract_features
    get_box_t = teacher.predict_boxes_logits
    get_box_s = student.predict_boxes_logits

    for step_idx, (img_weak, img_strong) in enumerate(tqdm(loader, desc=f"KDD ({kind}) train")):
        if step_idx >= max_iter:
            break

        xw = img_weak.to(device, non_blocking=True)
        xs = img_strong.to(device, non_blocking=True)

        loss_total = xs.new_zeros(())

        if kdd_cls is not None and w_cls > 0.0:
            with torch.no_grad():
                t_logits = teacher(xw)
            s_logits = student(xs)

            loss_kl, conf, w = kdd_cls(t_logits, s_logits, weight=None)
            loss_total = loss_total + w_cls * loss_kl

            if "kdd_kl" in history:
                history["kdd_kl"] += float(loss_kl.item())
            if "kdd_conf" in history:
                history["kdd_conf"] += float(conf.mean().item())
            if "kdd_w" in history:
                history["kdd_w"] += float(w.mean().item())

        if kdd_feat is not None and w_feat > 0.0:
            if get_feat_t is None or get_feat_s is None:
                raise RuntimeError("Requires feature extraction methods on teacher and student.")
            with torch.no_grad():
                f_t = get_feat_t(xw)
            f_s = get_feat_s(xs)

            loss_f = kdd_feat(f_t=f_t, f_s=f_s)
            loss_total = loss_total + w_feat * loss_f

            if "kdd_feat" in history:
                history["kdd_feat"] += float(loss_f.item())

        if kdd_box is not None and w_box > 0.0:
            if get_box_t is None or get_box_s is None:
                raise RuntimeError("Requires logits -> boxes, logits, valid masks methods on teacher and student.")
        
            with torch.no_grad():
                t_boxes, t_logits, t_valid = get_box_t(xw)
            s_boxes, s_logits, s_valid = get_box_s(xs)

            loss_b = kdd_box(t_boxes, t_logits, t_valid, s_boxes, s_logits, s_valid)
            loss_total = loss_total + w_box * loss_b
            if "kdd_box" in history:
                history["kdd_box"] += float(loss_b.item())

        optimizer.zero_grad(set_to_none=True)
        loss_total.backward()
        optimizer.step()
        scheduler.step()

        if "total" in history:
            history["total"] += float(loss_total.item())
        steps += 1

    return mean_history(history, steps)


def pipeline_kdd(
    cfg: ExperimentConfig,
    data: dict[str, DataLoader],
    device: torch.device,
    teacher_ckpt: str, student_ckpt: str,
    metric_keys: list[str], top_k: int
) -> None:
    teacher = build_model(cfg=cfg).to(device)
    student = build_model(cfg=cfg).to(device)

    teacher, _, _ = load_checkpoint(teacher_ckpt, teacher, optimizer=None, device=device)
    student, _, _ = load_checkpoint(student_ckpt, student, optimizer=None, device=device)

    for p in teacher.parameters():
        p.requires_grad_(False)

    optimizer = build_optimizer(cfg=cfg, model=student)
    steps_per_epoch = len(data["train_weak"])
    lr_scheduler = build_scheduler(
        optimizer=optimizer, scheme=cfg.sched.scheme, 
        total_epochs=cfg.train.epochs, steps_per_epoch=steps_per_epoch,
        warmup_epochs=cfg.sched.warmup_epochs, warmup_bias_lr=cfg.sched.warmup_bias_lr,
        min_lr_ratio=cfg.sched.min_lr_ratio, milestones=cfg.sched.milestones, gamma=cfg.sched.gamma)

    kind, weights, kdd_cls, kdd_feat, kdd_box = init_kdd(cfg, device)
    plotter = TrainingCurveSupervised(metrics=metric_keys)

    graphs_dir = "graphs"
    os.makedirs(graphs_dir, exist_ok=True)

    kl_kdd_curve: list[float] = []
    kl_teacher_curve: list[float] = []
    kl_ema_curve: list[float] = [] 

    for epoch in tqdm(range(cfg.train.epochs), desc=f"KDD epochs ({kind})"):
        train_hist = train_kdd_one_epoch(
            teacher=teacher, student=student,
            optimizer=optimizer, scheduler=lr_scheduler, data=data, device=device,
            max_iter=(cfg.train.log_interval * 999999), metric_keys=metric_keys,
            kind=kind, weights=weights, kdd_cls=kdd_cls, kdd_feat=kdd_feat, kdd_box=kdd_box)

        plotter.update(train_hist)
        plotter.plot_total(
            save_dir=graphs_dir, show=False,
            save_path=os.path.join(graphs_dir, f"kdd_{cfg.model.arch}_total.png"))

        b = next(iter(data["train_weak"]))
        img_weak, img_strong = b

        if torch.is_tensor(img_weak) and torch.is_tensor(img_strong):
            xw = img_weak[:min(64, img_weak.size(0))].to(device, non_blocking=True)
            xs = img_strong[:min(64, img_strong.size(0))].to(device, non_blocking=True)

            with torch.no_grad():
                get_logits_t = getattr(teacher, "predict_class_logits", None)
                get_logits_s = getattr(student, "predict_class_logits", None)
                t_logits = get_logits_t(xw) if callable(get_logits_t) else teacher(xw)
                s_logits = get_logits_s(xs) if callable(get_logits_s) else student(xs)

            p_t = softmax_mean(t_logits)
            p_s = softmax_mean(s_logits)
            kl_val = kl_divergence(p=p_t, q=p_s)
            kl_kdd_curve.append(float(kl_val))

            plot_kl_stagewise(
                kl_ema=kl_ema_curve, kl_teacher=kl_teacher_curve, kl_kdd=kl_kdd_curve, arch_name=str(cfg.model.arch), 
                show=False, save_path=os.path.join(graphs_dir, f"kdd_{cfg.model.arch}_kl_stagewise.png"))

            t_lab = torch.topk(t_logits, k=top_k, dim=1).indices.squeeze(1).detach().cpu().numpy()
            s_lab = torch.topk(s_logits, k=top_k, dim=1).indices.squeeze(1).detach().cpu().numpy()

            class_names = [str(i) for i in range(int(cfg.data.num_classes))]
            cm = agreement_matrix(
                teacher_labels=t_lab, student_labels=s_lab,
                num_classes=len(class_names), normalize=True)

            plot_agreement_heatmap(
                cm=cm, class_names=class_names, title=f"KDD agreement ({cfg.model.arch}) e{epoch + 1}",
                show=False, save_path=os.path.join(graphs_dir, f"kdd_{cfg.model.arch}_agree_e{epoch + 1}.png"))

            # EMA-vs-KDD plot API expects 4 label arrays; in KDD stage we only have teacher vs student,
            # so we pass the same pair into both panes to keep a consistent artifact across stages.
            plot_agreement_ema_vs_kdd(
                teacher_ema=t_lab, student_ema=s_lab,
                teacher_kdd=t_lab, student_kdd=s_lab,
                class_names=class_names, arch_name=str(cfg.model.arch),
                show=False, save_path=os.path.join(graphs_dir, f"kdd_{cfg.model.arch}_ema_vs_kdd_e{epoch + 1}.png"))

            # Cross-arch / cross-dataset plots expect dict curves;
            # in single-run we provide one entry
            plot_cross_arch_kl(
                kl_by_arch={str(cfg.model.arch): kl_kdd_curve}, teacher_arch=str(cfg.model.arch),
                show=False, save_path=os.path.join(graphs_dir, f"kdd_{cfg.model.arch}_cross_arch_kl.png"))

            plot_cross_dataset_kl(
                kl_by_dataset={str(cfg.data.dataset): kl_kdd_curve},
                teacher_dataset=str(cfg.data.dataset), teacher_arch=str(cfg.model.arch),
                show=False, save_path=os.path.join(graphs_dir, f"kdd_{cfg.model.arch}_cross_dataset_kl.png"))

        if (epoch + 1) % cfg.train.ckpt_interval == 0 or (epoch + 1) == cfg.train.epochs:
            train_set = getattr(data["train_weak"], "dataset", None)

            # For Grad-CAM models, perform bbox evaluation at each checkpoint
            if cfg.model.arch == "resnet_gradcam":
                metrics: dict[str, float] = {}
                if hasattr(data["val"], "dataset"):
                    val_set = data["val"].dataset
                    metrics = evaluate_cam_bboxes(
                        campp=student.campp, model=student.backbone,
                        images=val_set.images.to(device, non_blocking=True),
                        gt_boxes=val_set.gt_boxes.to(device, non_blocking=True),
                        gt_labels=val_set.gt_labels.to(device, non_blocking=True),
                        iou_thr=0.5, cam_thr=0.35)
                print(f"KDD epoch {epoch + 1} Grad-CAM evaluation metrics: {metrics}")

            visualize_weight_distribution(
                student, out_dir=graphs_dir,
                file_name=f"kdd_{cfg.model.arch}_weights_e{epoch + 1}.png")

            if train_set is not None:
                visualize_gradients(
                    student, train_set=train_set, device=device,
                    batch_size=min(256, cfg.data.batch_size),
                    out_dir=graphs_dir, file_name=f"kdd_{cfg.model.arch}_grads_e{epoch + 1}.png")
                visualize_activations(
                    student, train_set=train_set, device=device,
                    batch_size=min(256, cfg.data.batch_size),
                    out_dir=graphs_dir, file_name=f"kdd_{cfg.model.arch}_acts_e{epoch + 1}.png")

            os.makedirs("kdd" + cfg.model.arch + "_checkpoints", exist_ok=True)
            ckpt_path = os.path.join("kdd" + cfg.model.arch + "_checkpoints", f"checkpoint_epoch_{epoch + 1}.pth")
            save_checkpoint(student, optimizer, epoch + 1, ckpt_path)
