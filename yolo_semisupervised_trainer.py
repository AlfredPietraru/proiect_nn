from __future__ import annotations
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import warnings
warnings.filterwarnings("ignore")

from models.ema import EMA

import os
from utils import Logger
from ultralytics.models import YOLO
# from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.models.yolo.detect.train import DetectionTrainer

import torch
from copy import deepcopy
from ultralytics.utils.torch_utils import ModelEMA

class SemiSupervisedTrainer(DetectionTrainer):
    def __init__(self, *args, ssl_config=None, **kwargs):
        """
        Extends DetectionTrainer to support semi‑supervised training:
            - student + teacher models
            - two dataloaders (labeled and unlabeled)
            - supervised + unsupervised loss
        Args:
            ssl_config (dict): {
                "unlabeled_data": path/to/unlabeled.yaml,
                "unsup_weight": float,
                "pseudo_conf_thr": float,
                "ema_decay": float
            }
        """
        self.ssl_config = ssl_config or {}
        super().__init__(*args, **kwargs)

    def build_unlabeled_dataloader(self):
        """
        Build an unlabeled dataloader using the same logic as DetectionTrainer.get_dataloader().
        """
        unlabeled_cfg = self.ssl_config.get("unlabeled_data")
        return self.get_dataloader(unlabeled_cfg, batch_size=self.batch_size, rank=0, mode="train")

    def _ss_train_step(self, labeled_batch, unlabeled_batch):
        """
        Perform one semi‑supervised training step:
            - supervised loss
            - teacher generate pseudo labels for unlabeled
            - student unsupervised loss
        """
        # Move images to device & normalize to [0,1]
        imgs_l = labeled_batch["img"].to(self.device).float() / 255.0
        imgs_u = unlabeled_batch["img"].to(self.device).float() / 255.0

        # 1) Supervised loss
        sup_preds = self.student(imgs_l)
        sup_loss_vec, _ = self.criterion(sup_preds, labeled_batch)
        sup_loss = sup_loss_vec.sum()

        # 2) Teacher pseudo labels (no gradients)
        with torch.no_grad():
            teacher_outputs = self.teacher.ema(imgs_u)
            pseudo_targets = self.generate_pseudo_labels(teacher_outputs)

        # 3) Student unsupervised loss
        unsup_preds = self.student(imgs_u)
        unsup_loss_vec, _ = self.criterion(unsup_preds, pseudo_targets)
        unsup_loss = unsup_loss_vec.sum()

        # 4) Combined loss
        total_loss = sup_loss + self.unsup_weight * unsup_loss
        return total_loss, sup_loss, unsup_loss

    def generate_pseudo_labels(self, teacher_outputs):
        """
        Create pseudo labels in the same target dict format YOLOv8 expects,
        filtering by confidence threshold.
        """
        imgs = teacher_outputs[0] if isinstance(teacher_outputs, tuple) else teacher_outputs
        batch_size = len(imgs)

        boxes_list = []
        cls_list = []
        idx_list = []

        for b in range(batch_size):
            out = imgs[b]
            # apply filtering mask on scores
            keep = out["scores"] >= self.pseudo_conf_thr
            boxes = out["boxes"][keep]
            cls   = out["labels"][keep]
            batch_idx = torch.full((boxes.shape[0], 1), b, dtype=torch.long, device=self.device)

            boxes_list.append(boxes)
            cls_list.append(cls)
            idx_list.append(batch_idx)

        return {
            "bboxes": torch.cat(boxes_list, dim=0),
            "cls": torch.cat(cls_list, dim=0),
            "batch_idx": torch.cat(idx_list, dim=0)
        }

    def _do_train(self):
        """
        Override the main training loop to use annotated + unlabeled data.
        """
        self.train_loader = self.get_dataloader(self.data["train"], self.batch_size, rank=0, mode="train")
        self.unlabeled_loader = self.build_unlabeled_dataloader()

        unlabeled_iter = iter(self.unlabeled_loader)
        self.optimizer.zero_grad()

        for i, labeled_batch in enumerate(self.train_loader):
            try:
                unlabeled_batch = next(unlabeled_iter)
            except StopIteration:
                unlabeled_iter = iter(self.unlabeled_loader)
                unlabeled_batch = next(unlabeled_iter)

            with torch.cuda.amp.autocast(self.amp):
                total_loss, sup_loss, unsup_loss = self._ss_train_step(labeled_batch, unlabeled_batch)

            self.scaler.scale(total_loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()

            # EMA update of teacher
            self.teacher.update(self.student)

            # Logging
            if i % 10 == 0:
                print(f"[Batch {i}] sup={sup_loss.item():.4f} unsup={unsup_loss.item():.4f} total={total_loss.item():.4f}")

        # After the epoch, optionally run validation
        self.validate()
