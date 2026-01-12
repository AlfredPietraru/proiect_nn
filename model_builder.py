from models.hyperparams import ExperimentConfig
import torch
from models.faster_resnet import get_model_fasterrcnn
from models.gradcam_resnet import ResNet50GradCamPP
from models.yolon11 import get_model_yolo11

def build_model(cfg: ExperimentConfig) -> torch.nn.Module:
    """
    Models used in the experiments are only 3:
    (make sure that the model.arch in config is one of these)
    - Faster R-CNN with ResNet50-FPN backbone
    - ResNet50 with GradCAM++
    - YOLO-N11
    """
    arch = getattr(cfg.model, "arch")
    if arch == "fasterrcnn":
        return get_model_fasterrcnn(cfg=cfg)
    if arch == "resnet50_gradcampp":
        return ResNet50GradCamPP(cfg=cfg)
    if arch == "yolo11":
        return get_model_yolo11(cfg=cfg)
    raise ValueError(f"Unknown Model Architecture: {arch}")