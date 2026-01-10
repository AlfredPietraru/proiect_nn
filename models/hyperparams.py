from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

DatasetName = Literal["voc", "visdrone", "uavdt", "auair"]
ArchName = Literal["fasterrcnn", "resnet50_gradcampp", "yolo11n"]
OptName = Literal["sgd", "adamw", "adam"]
SchedName = Literal["cosine", "multistep"]
KDDKind = Literal["weakstrong", "cross_dataset", "feature", "box_match", "combo"]


def dataset_num_classes(dataset: str) -> int:
    """
    Based on the dataset details found online.
    Each dataset may have varying number of classes,
    these are the actual numbers for each dataset.
    """
    ds = dataset.lower()
    if ds == "voc":
        return 20
    if ds == "visdrone":
        return 10
    if ds == "uavdt":
        return 4
    if ds == "auair":
        return 8
    raise ValueError(f"Unknown dataset='{dataset}'")


def dataset_max_objects(dataset: str) -> int:
    """
    Based on the dataset details found online.
    Each dataset may have images with varying number of objects,
    so these are approximate upper bounds.
    """
    ds = dataset.lower()
    if ds == "voc":
        return 30
    if ds == "visdrone":
        return 300
    if ds == "uavdt":
        return 100
    if ds == "auair":
        return 100
    raise ValueError(f"Unknown dataset='{dataset}'")


@dataclass
class DataCfg:
    dataset: DatasetName = "voc"
    root: str = "datasets"
    percentage = 0.05

    voc_dir: str = "VOC"
    visdrone_dir: str = "VisDrone"
    uavdt_dir: str = "UAVDT"
    auair_dir: str = "AUAIR"

    img_size: int = 512
    batch_size: int = 8
    num_workers: int = 4
    pin_memory: bool = True
    download: bool = False

    labeled_percent: float = 0.10
    unsup_batch_ratio: float = 1.0
    use_strong_aug: bool = True

    num_classes: int = 20
    max_objects: int = 300

    def sync(self) -> None:
        self.num_classes = int(dataset_num_classes(self.dataset))
        self.max_objects = 300


@dataclass
class ModelCfg:
    arch: ArchName = "fasterrcnn"
    num_classes: int = 20

    pretrained: bool = True
    freeze_backbone: bool = False

    yolo_weights: str = "yolo11n.pt"
    yolo_iou: float = 0.7
    yolo_agnostic_nms: bool = True
    yolo_max_det: int = 300

    def sync(self, data: DataCfg) -> None:
        self.num_classes = int(data.num_classes)
        self.yolo_max_det = int(data.max_objects)


@dataclass
class OptimCfg:
    opt: OptName = "sgd"
    lr: float = 0.01
    weight_decay: float = 1e-4
    momentum: float = 0.9
    nesterov: bool = True
    betas: tuple[float, float] = (0.9, 0.999)
    head_lr_mult: float = 1.0


@dataclass
class SchedCfg:
    scheme: SchedName = "cosine"
    warmup_epochs: int = 3
    warmup_bias_lr: float = 0.1
    min_lr_ratio: float = 0.05

    milestones: list[int] = field(default_factory=lambda: [30, 50])
    gamma: float = 0.1


@dataclass
class TrainCfg:
    device: str = "cuda:0"
    epochs: int = 10
    use_amp: bool = True
    max_grad_norm: float | None = None

    log_interval: int = 10
    eval_interval: int = 1
    ckpt_interval: int = 1


@dataclass
class SSLTrainCfg:
    burnin_epochs: int = 1
    unsup_start_epoch: int = 1

    sup_weight: float = 1.0
    unsup_weight: float = 4.0

    use_teacher: bool = True
    ema_decay: float = 0.9996

    pseudo_conf_thr: float = 0.7
    match_iou_thr: float = 0.5
    max_pairs_per_image: int = 128

    strong_aug_on_student: bool = True


@dataclass
class KDDCfg:
    kind: KDDKind = "weakstrong"

    # weights (used by kind="combo"; also safe for single-kind)
    w_cls: float = 1.0
    w_feat: float = 0.0
    w_box: float = 0.0

    # KL knobs
    tau: float = 2.0
    gamma: float = 0.7
    eps: float = 0.0

    # BoxMatchKDD knobs
    iou_thr: float = 0.5
    box_l1: float = 0.0

    # FeatureKDD knobs
    beta: float = 1.0

    # Cross-dataset class mapping (teacher class id -> student class id)
    # Only required for kind="cross_dataset"
    teacher_to_student: Optional[dict[int, int]] = None


@dataclass
class MetricsCfg:
    num_classes: int = 20
    score_thr: float = 0.35
    class_agnostic: bool = False
    iou_thrs: tuple[float, ...] = (
        0.5, 0.55, 0.6, 0.65, 0.7,
        0.75, 0.8, 0.85, 0.9, 0.95,
    )

    def sync(self, data: DataCfg) -> None:
        self.num_classes = int(data.num_classes)


@dataclass
class ExperimentConfig:
    seed: int = 42

    data: DataCfg = field(default_factory=DataCfg)
    model: ModelCfg = field(default_factory=ModelCfg)
    optim: OptimCfg = field(default_factory=OptimCfg)
    sched: SchedCfg = field(default_factory=SchedCfg)
    train: TrainCfg = field(default_factory=TrainCfg)
    ssl: SSLTrainCfg = field(default_factory=SSLTrainCfg)
    kdd: KDDCfg = field(default_factory=KDDCfg)
    metrics: MetricsCfg = field(default_factory=MetricsCfg)

    def __post_init__(self) -> None:
        self.sync()

    def sync(self) -> None:
        self.data.sync()
        self.model.sync(self.data)
        self.metrics.sync(self.data)
        self.train.device = str(self.train.device)

    def num_classes_with_bg(self) -> int:
        return int(self.data.num_classes) + 1
