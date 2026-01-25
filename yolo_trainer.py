from __future__ import annotations
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import warnings
warnings.filterwarnings("ignore")

from models.ema import EMA

import os
import torch
from utils import Logger
from ultralytics.models import YOLO
# from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.loss import v8DetectionLoss
# from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.models.yolo.detect.predict import DetectionPredictor
import yaml 
from ultralytics.data.dataset import YOLODataset
from ultralytics.data.build import build_dataloader, build_yolo_dataset
from yolo_semisupervised_trainer import SemiSupervisedTrainer


def do_burn_in_step():
    model = YOLO()
    model.train(
        data='VOC.yaml',
        epochs=3,
        batch=16,
        imgsz=640,
        lr0=0.01,               
        warmup_epochs=3,        
        device='cuda',
        fraction=0.5
    )

if __name__ == "__main__":
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Load dataset config
    with open("VOC.yaml", "r") as f:
        data_cfg = yaml.load(f, Loader=yaml.SafeLoader)

    # Build labeled data loader
    labeled_loader = build_dataloader(
        dataset=YOLODataset(
            img_path="datasets/VOC/images/train2007", fraction=0.3, data=data_cfg, task="detect"),
        batch=8, workers=4, shuffle=True, pin_memory=False)

    unlabeled_loader = build_dataloader(
        dataset=YOLODataset(img_path="datasets/VOC/images/train2012", fraction=0.3, data=data_cfg, task="detect"),
        batch=8, workers=4, shuffle=True, pin_memory=False)


    student_yolo = YOLO("yolo11n.pt").to(DEVICE)
    student_yolo.model.train()        
    ema = EMA(teacher_model=student_yolo.model, decay=0.9996, init_from=student_yolo.model, adaptive=True)
    teacher = ema.model()
    teacher.eval()
    optimizer = torch.optim.Adam(student_yolo.model.parameters(), lr=1e-4)
    batch = next(iter(labeled_loader))

    imgs = batch["img"].to(DEVICE).float() / 255.0
    targets = {
        "bboxes": batch["bboxes"].to(DEVICE),
        "cls": batch["cls"].to(DEVICE),
        "batch_idx": batch["batch_idx"].to(DEVICE),
    }

    optimizer.zero_grad()
    preds = student_yolo(imgs, targets)
    criterion = v8DetectionLoss(student_yolo.model)
    loss, loss_items = criterion(preds, targets)  # this is a differentiable tensor
    total_loss = loss.sum()
    print("requires_grad:", total_loss.requires_grad)

    # total_loss = student_yolo(imgs, targets)

    # print("requires_grad:", total_loss.requires_grad)

    total_loss.backward()
    optimizer.step()

    