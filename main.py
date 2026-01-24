from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import os  # noqa: E402
import torch  # noqa: E402
from typing import cast  # noqa: E402

from src.utils import Logger, set_seed  # noqa: E402

from src.data import build_dataloaders  # noqa: E402
from src.dataset_details import dataset_details  # noqa: E402

from src.burn_in import pipeline_burn_in  # noqa: E402
from src.unbiased_teacher import pipeline_semi_supervised  # noqa: E402
from src.kl_divergence import pipeline_kdd  # noqa: E402

from src.models.hyperparams import ExperimentConfig, ArchName, dataset_classes, kdd_metric_keys  # noqa: E402


# Training loss keys to LOG (not eval metrics like mAP/IoU)
# Must match keys returned by each model's forward(..., targets) loss dict.

# Faster R-CNN (TorchVision losses)
# Returned keys come from rpn_losses + roi_losses; we add "total" in our wrapper.
FRCNN_BURNIN = [
    "loss_classifier",   # ROI head class loss (label classification)
    "loss_box_reg",      # ROI head bbox regression loss (proposal -> GT refinement)
    "loss_objectness",   # RPN objectness loss (foreground/background for anchors)
    "loss_rpn_box_reg",  # RPN bbox regression loss (anchor -> proposal refinement)
    "total"              # sum of all the above (added in FasterRCNNResNet50FPN.forward)
]
FRCNN_SUP   = FRCNN_BURNIN     # same keys during labeled part of SSL
FRCNN_UNSUP = FRCNN_BURNIN     # same keys during pseudo-labeled part of SSL


# GradCAM++ ResNet50 (classifier CE + CAM boxes)
# This model is not a native detector; its training loss is classification CE only.
# The CAM boxes are evaluated separately (IoU/recall/hit-rate) via gradcam_eval.py.
GCAM_BURNIN = [
    "loss",   # cross-entropy classification loss
    "total"   # alias to loss (kept to match other pipelines)
]
GCAM_SUP    = GCAM_BURNIN       # same keys during labeled part of SSL
GCAM_UNSUP  = GCAM_BURNIN       # same keys during pseudo-labeled part of SSL


# YOLOv11 (Ultralytics detect loss components)
# In our wrapper we expose semantic components:
# - loss_box, loss_cls, loss_dfl (when loss_items has >= 3 entries)
YOLO_BURNIN = [
    "loss",      # total YOLO loss used for backprop
    "total",     # alias to loss
    "loss_box",  # box regression component
    "loss_cls",  # classification component
    "loss_dfl"   # distribution focal loss component (bbox distribution)
]
YOLO_SUP    = YOLO_BURNIN       # same keys during labeled part of SSL
YOLO_UNSUP  = YOLO_BURNIN       # same keys during pseudo-labeled part of SSL


def metric_pack_for_arch(arch: str):
    a = arch.lower()
    if a == "fasterrcnn":
        return FRCNN_BURNIN, FRCNN_SUP, FRCNN_UNSUP
    if a == "resnet50_gradcampp":
        return GCAM_BURNIN, GCAM_SUP, GCAM_UNSUP
    if a in ("yolo11n", "yolo11"):  # tolerate both
        return YOLO_BURNIN, YOLO_SUP, YOLO_UNSUP
    raise ValueError(f"Unknown arch: {arch}")


def ckpt_dir(arch: str) -> str:
    d = "model_" + str(arch) + "_checkpoints"
    os.makedirs(d, exist_ok=True)
    return d


def ckpt_path(arch: str, epoch: int) -> str:
    return os.path.join(ckpt_dir(arch), f"checkpoint_epoch_{int(epoch)}.pth")


def run_experiment(cfg: ExperimentConfig, logger: Logger, teacher_arch: ArchName, student_arch: ArchName) -> tuple[str, str]:
    """
    Runs: Burn-in -> SSL -> KDD pipeline.
    Teacher and student architectures can be the same or different.
    Returns the teacher and student SSL checkpoint paths.
    """
    device = torch.device(cfg.train.device)

    # Finding different details about the dataset
    # Like: distribution of classes, t-SNE plots of image embeddings,
    # statistics mean and std, distribution of bounding box sizes, etc.
    classes = dataset_classes(cfg.data.dataset)

    cfg.model.arch = teacher_arch
    data_teacher = build_dataloaders(cfg)
    dataset_details(
        data_teacher, classes, seed=cfg.seed,
        save_path=os.path.join("dataset_details", cfg.model.arch),
        max_batches=60, max_images=3000, embed_hw=(32, 32), show=False,
        tsne_perplexity=30.0, tsne_iter=3000)

    metric_burnin, metric_sup, metric_unsup = metric_pack_for_arch(cfg.model.arch)

    # Burn-in phase for supervised pretraining
    # Using only the labeled data for training with strong augmentations
    # After this phase, we will have a supervised model checkpoint
    pipeline_burn_in(cfg=cfg, data=data_teacher, device=device, metric_keys=metric_burnin)
    burnin_ckpt_t = ckpt_path(cfg.model.arch, cfg.train.epochs)

    # Semi-supervised phase using Unbiased Teacher
    # Using both labeled and unlabeled data for training with teacher-student paradigm
    # Paradigm uses pseudo-labeling and consistency regularization and EMA (Exponential Moving Average)
    # EMA helps to stabilize the teacher model predictions over time, can work as a form of regularization
    # and even with different augmentations of the same image, the teacher model can provide consistent pseudo-labels
    # and with different architectures for teacher and student models, the student can learn complementary features
    # After this phase, we will have a semi-supervised model checkpoint
    pipeline_semi_supervised(cfg, burnin_ckpt_t, data_teacher, device, metric_sup=metric_sup, metric_unsup=metric_unsup)
    ssl_ckpt_t = ckpt_path(cfg.model.arch, cfg.train.epochs)

    student_ssl_ckpt = ssl_ckpt_t
    data_student = data_teacher

    if student_arch is not None and str(student_arch).lower() != str(teacher_arch).lower():
        cfg.model.arch = student_arch
        data_student = build_dataloaders(cfg)
        dataset_details(
            data_student, classes, seed=cfg.seed,
            save_path=os.path.join("dataset_details", cfg.model.arch),
            max_batches=60, max_images=3000, embed_hw=(32, 32), show=False,
            tsne_perplexity=30.0, tsne_iter=3000)

        metric_burnin, metric_sup, metric_unsup = metric_pack_for_arch(cfg.model.arch)

        # Burn-in phase for supervised pretraining
        # Using only the labeled data for training with strong augmentations
        # After this phase, we will have a supervised model checkpoint
        pipeline_burn_in(cfg=cfg, data=data_student, device=device, metric_keys=metric_burnin)
        burnin_ckpt_s = ckpt_path(cfg.model.arch, cfg.train.epochs)

        # Semi-supervised phase using Unbiased Teacher
        # Using both labeled and unlabeled data for training with teacher-student paradigm
        # Paradigm uses pseudo-labeling and consistency regularization and EMA (Exponential Moving Average)
        # EMA helps to stabilize the teacher model predictions over time, can work as a form of regularization
        # and even with different augmentations of the same image, the teacher model can provide consistent pseudo-labels
        # and with different architectures for teacher and student models, the student can learn complementary features
        # After this phase, we will have a semi-supervised model checkpoint
        pipeline_semi_supervised(cfg, burnin_ckpt_s, data_student, device, metric_sup=metric_sup, metric_unsup=metric_unsup)
        student_ssl_ckpt = ckpt_path(cfg.model.arch, cfg.train.epochs)

    # Knowledge Distillation phase using KDD
    # Using both labeled and unlabeled data for training with teacher-student paradigm
    # The teacher model is the one obtained after semi-supervised training
    # The student model is initialized from the same checkpoint as the teacher
    # The student model learns from the soft labels provided by the teacher model
    # Soft labels contain more information about the uncertainty of the predictions
    # This phase helps to compress the model and improve generalization
    # After this phase, we will have a distilled model checkpoint
    metric_kdd = kdd_metric_keys(cfg.kdd.kind)
    pipeline_kdd(cfg, data_student, device, teacher_ckpt=ssl_ckpt_t, student_ckpt=student_ssl_ckpt, metric_keys=metric_kdd, top_k=cfg.kdd.top_k)

    logger.info(f"[DONE] teacher={teacher_arch} student={student_arch or teacher_arch}")
    return ssl_ckpt_t, student_ssl_ckpt


if __name__ == "__main__":
    cfg = ExperimentConfig()
    cfg.sync()
    set_seed(cfg.seed)

    logger = Logger("PIPELINE", log_dir="logs", rich_tracebacks=False)

    # Run all 3 models sequentially on same dataset/config
    # Note: hyperparams ArchName includes yolo11n, builders may use yolo11.
    # We'll prefer yolo11n but tolerate both in metric_pack_for_arch.
    for arch in ["fasterrcnn", "resnet50_gradcampp", "yolo11n"]:
        cfg.model.arch = cast(ArchName, arch)
        logger.info(f"RUN: dataset={cfg.data.dataset} arch={cfg.model.arch} kdd={cfg.kdd.kind}")
        logger.info("=" * 90)
        run_experiment(cfg, logger, teacher_arch=cast(ArchName, arch), student_arch=cast(ArchName, arch))

    # Cross-architecture student-teacher (optional)
    # Burn-in -> SSL for TEACHER arch, Burn-in -> SSL for STUDENT arch, then KDD(teacher_ckpt -> student_ckpt).
    # Keep cfg.kdd.kind consistent with the compatibility you implemented.
    # Example pairs:
    # run_experiment(cfg, logger, teacher_arch=cast(ArchName, "fasterrcnn"), student_arch=cast(ArchName, "yolo11n"))
    # run_experiment(cfg, logger, teacher_arch=cast(ArchName, "yolo11n"), student_arch=cast(ArchName, "fasterrcnn"))

    logger.info("\nALL EXPERIMENTS FINISHED.\n")
