from __future__ import annotations
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import warnings
warnings.filterwarnings("ignore")

import os
import torch

from utils.oncuda import set_seed
from data import build_dataloaders, download_all_datasets
from dataset_details import dataset_details
from burn_in import pipeline_burn_in
from unbiased_teacher import pipeline_semi_supervised
from kdd import pipeline_kdd
from models.hyperparams import ExperimentConfig, dataset_classes
from utils import Logger


if __name__ == "__main__":
    cfg = ExperimentConfig()
    set_seed(cfg.seed)
    
    # Guarantee datasets are downloaded and available
    logger = Logger("PIPELINE", log_dir="logs", rich_tracebacks=False)
    # download_all_datasets(details=logger)

    device = torch.device(cfg.train.device)
    checkpoint_dir = "model_" + cfg.model.arch + "_checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    data = build_dataloaders(cfg)

    # Finding different details about the dataset
    # Like: distribution of classes, t-SNE plots of image embeddings,
    # statistics mean and std, distribution of bounding box sizes, etc.
    classes = dataset_classes(cfg.data.dataset)
    dataset_details(
        data, classes,
        seed=cfg.seed, save_path="dataset_details", 
        max_batches=60, max_images=3000, embed_hw=(32, 32), 
        tsne_perplexity=30.0, tsne_iter=3000, show=False)

    #! UPDATE HERE THE METRICS YOU WANT TO TRACK DURING TRAINING
    METRIC_BURN_IN = ["loss_classifier", "loss_bbox", "loss_objectness"]
    METRIC_SUP = ["loss_classifier", "loss_bbox", "loss_objectness"]
    METRIC_UNSUP = ["loss_classifier", "loss_objectness"]

    # Burn-in phase for supervised pretraining
    # Using only the labeled data for training
    # After this phase, we will have a supervised model checkpoint
    pipeline_burn_in(cfg=cfg, data=data, device=device, metric_keys=METRIC_BURN_IN)
    last_ckpt = os.path.join(checkpoint_dir, f"checkpoint_epoch_{cfg.train.epochs}.pth")

    # Semi-supervised phase using Unbiased Teacher approach
    # Using both labeled and unlabeled data for training with teacher-student paradigm
    # Paradigm uses pseudo-labeling and consistency regularization and EMA (Exponential Moving Average)
    # EMA helps to stabilize the teacher model predictions over time, can work as a form of regularization
    # and even with different augmentations of the same image, the teacher model can provide consistent pseudo-labels
    # and with different architectures for teacher and student models, the student can learn complementary features
    # After this phase, we will have a semi-supervised model checkpoint

    pipeline_semi_supervised(cfg, last_ckpt, data, device, metric_sup=METRIC_SUP, metric_unsup=METRIC_UNSUP)
    # Knowledge Distillation phase using KDD approach
    # Using both labeled and unlabeled data for training with teacher-student paradigm
    # The teacher model is the one obtained after semi-supervised training
    # The student model is initialized from the same checkpoint as the teacher
    # The student model learns from the soft labels provided by the teacher model
    # Soft labels contain more information about the uncertainty of the predictions
    # This phase helps to compress the model and improve generalization
    # After this phase, we will have a distilled model checkpoint
    pipeline_kdd(cfg, data, device, teacher_ckpt=last_ckpt, student_ckpt=last_ckpt, metric_keys=METRIC_SUP)
