import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from voc import get_dataloader
from main_utils import set_seed
from model_factory import get_model
from ema import RobustEMA
import os
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from box_ops import BoxList
from metrics import DetectionMetrics
from config_params import Metrics
from torchvision.ops import batched_nms
from main_utils import (
    save_checkpoint, load_checkpoint, plot_losses, plot_validation_results
)

set_seed(42)
SIZE = (224, 224)
CONFIDENCE_THRESHOLD = 0.7
BATCH_SIZE = 2
CHECKPOINT_DIR = "./checkpoints"
METRIC_SUPERVISED = ["loss_classifier", "loss_box_reg", "loss_objectness", "loss_rpn_box_reg"]
METRICS_UNSUPERVISED = ["loss_classifier", "loss_objectness"]
VALIDATION_METRICS = ["mAP_50", "mAP_5095", "precision", "recall", "f1"]
LAMBDA_UNSUPERVISED = 5.0
NMS_IOU = 0.5
ITERATION_TO_STOP_AT = 5000

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

def scale_to_01(image, **kwargs):
    return image.astype('float32') / 255.0

weak_augmentations = A.Compose([
    A.Resize(SIZE[0], SIZE[1]),         
    A.HorizontalFlip(p=0.5),
    A.Lambda(image=scale_to_01), 
    ToTensorV2(),
], bbox_params=A.BboxParams(format='pascal_voc'))

strong_augmentations = A.Compose(
        [
            A.Resize(SIZE[0], SIZE[1]),
            A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=0.8),
            A.GaussianBlur(blur_limit=(3, 7), sigma_limit=(0.1, 2.0), p=0.5),
            A.CoarseDropout(num_holes_range=(3, 3), hole_height_range=(0.05, 0.1),
                             hole_width_range=(0.05, 0.1), p=0.5),
            A.Lambda(image=scale_to_01), 
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(format='pascal_voc')
    )

test_transforms = A.Compose([
    A.Resize(SIZE[0], SIZE[1]),
    A.Lambda(image=scale_to_01), 
    ToTensorV2(), 
], bbox_params=A.BboxParams(format='pascal_voc'))


dt_train_labeled = get_dataloader("trainval", "2007", BATCH_SIZE, transform=weak_augmentations)
dt_train_unlabeled_weakaug = get_dataloader("trainval", "2012", BATCH_SIZE, transform=weak_augmentations) 
dt_train_unlabeled_strongaug = get_dataloader("trainval", "2012", BATCH_SIZE, transform=strong_augmentations)
dt_test = get_dataloader("test", "2007", BATCH_SIZE, transform=test_transforms, shuffle=False)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def train_burn_in(model, optimizer, dt_train_labeled, device):
    model.train()
    train_batches = 0
    history = {key : 0 for key in METRIC_SUPERVISED}

    for images, targets in tqdm(dt_train_labeled, desc="Training"):
        # if train_batches == 5: break
        for target in targets:
            target["boxes"] = target["boxes"].to(device)
            target["labels"] = target["labels"].to(device)
        images = images.to(device)
        loss_dict = model(images, targets)
        loss = sum(loss_dict.values())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        for k, v in loss_dict.items():
            history[k] += v.item()

        history["total"] += loss.item()
        train_batches += 1
    for key in history:
        history[key] = history[key] / train_batches
    return history


def pipeline_burn_in(epochs, dt_train_labeled, device, checkpoint_every):
    model = get_model(device=device)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    history = {key : [] for key in METRIC_SUPERVISED}

    for epoch in range(epochs):
        print(f"\n==================== Epoch {epoch+1}/{epochs} ====================\n")
        train_history = train_burn_in(model, optimizer, dt_train_labeled, device)
        lr_scheduler.step(train_history["total"])
        for key, val in train_history.items():
            history[key].append(val)
        plot_losses(history)
        if (epoch + 1) % checkpoint_every == 0 or (epoch + 1) == epochs:
            checkpoint_path = os.path.join(CHECKPOINT_DIR, f"checkpoint_epoch_{epoch+1}.pth")
            save_checkpoint(model, optimizer, epoch + 1, checkpoint_path)


def generate_pseudo_labels(model : torch.nn.Module, images : torch.Tensor, device):
    model.eval()
    with torch.no_grad():
        images = images.to(device)
        outputs = model(images, None)
        for output in outputs:
            boxes  = output["boxes"]
            labels = output["labels"]
            scores = output["scores"]

            keep_nms = batched_nms(
                boxes, scores, labels,
                iou_threshold=NMS_IOU
            )
            boxes  = boxes[keep_nms]
            labels = labels[keep_nms]
            scores = scores[keep_nms]

            boxes_to_keep = scores > CONFIDENCE_THRESHOLD        
            boxes  = boxes[boxes_to_keep]
            labels = labels[boxes_to_keep]
            scores = scores[boxes_to_keep]

            output["boxes"]  = boxes
            output["labels"] = labels
            output["scores"] = scores
        return outputs       
    

def train_semi_supervised_one_epoch(teacher : RobustEMA, student, optimizer, dt_labeled, dt_weak, dt_strong):
    student.train()
    train_batches = 0
    history = {}
    for key in METRIC_SUPERVISED:
        history[f"{key}_supervised"] = 0
    for key in METRICS_UNSUPERVISED:
        history[f"{key}_unsupervised"] = 0
    history["total"] = 0

    for (img_labeled, targets_labeled), (img_weak, _), (img_strong, _) in zip(dt_labeled, dt_weak, dt_strong):
        if train_batches == ITERATION_TO_STOP_AT: break
        print(train_batches)
        # SHOULD REPLACE THE TRANSFORMATION OF HORIZONTAL FLIP WITH SOMETHING PHOTOMETRIC
        weak_targets = generate_pseudo_labels(teacher.ema, img_weak, device)
        
        for target in weak_targets:
            target["boxes"] = target["boxes"].to(device)
            target["labels"] = target["labels"].to(device)
        img_strong = img_strong.to(device)
        loss_dict_unsupervised = student(img_strong, weak_targets)

        for target in targets_labeled:
            target["boxes"] = target["boxes"].to(device)
            target["labels"] = target["labels"].to(device)
        img_labeled = img_labeled.to(device)
        loss_dict_supervised = student(img_labeled, targets_labeled)

        optimizer.zero_grad()
        loss = sum(loss_dict_supervised.values()) + LAMBDA_UNSUPERVISED * (loss_dict_unsupervised["loss_classifier"] + loss_dict_unsupervised["loss_objectness"])
        loss.backward()
        optimizer.step()

        teacher.update(student)
        for k in METRICS_UNSUPERVISED:
            history[f"{k}_unsupervised"] += loss_dict_unsupervised[k].item()
        for k in METRIC_SUPERVISED:
            history[f"{k}_supervised"] += loss_dict_supervised[k].item()

        history["total"] += loss.item()
        train_batches += 1
    for key in history:
        history[key] = history[key] / train_batches
    return history


def validate_semi_supervised(student, dt_test, device, cfg_metrics : Metrics):
    student.eval()
    metrics = DetectionMetrics(cfg_metrics)
    metrics.reset()
    
    with torch.no_grad():
        for idx, (images, targets) in enumerate(dt_test):
            if idx == ITERATION_TO_STOP_AT: break
            print(idx)
            for target in targets:
                target["boxes"] = target["boxes"].to(device)
                target["labels"] = target["labels"].to(device)
            images = images.to(device)
            outputs = student(images, None)
            preds_bl = [BoxList(o["boxes"], o["labels"], o.get("scores", None), (images[0].shape[1], images[0].shape[2])) for o in outputs]
            tgts_bl = [BoxList(t["boxes"], t["labels"], t.get("scores", None), (images[0].shape[1], images[0].shape[2])) for t in targets]
            metrics.update(preds_bl, tgts_bl)

    
    student.train()
    loss = 0
    for idx, (images, targets) in enumerate(dt_test):
        if idx == ITERATION_TO_STOP_AT: break
        print(idx)
        for target in targets:
            target["boxes"] = target["boxes"].to(device)
            target["labels"] = target["labels"].to(device)
        images = images.to(device)
        loss_dict = student(images, targets)
        loss += sum(loss_dict.values()).item()

    metrics_dict = metrics.compute()  
    metrics_dict["validation_loss"] = loss / max(1, len(dt_test)) 
    return metrics_dict
    

class EarlyStopper:
    def __init__(self, patience=8, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0
        self.should_stop = False

    def step(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
        
        if self.counter >= self.patience:
            self.should_stop = True

def run_semi_supervised_pipeline(checkpoint_path, epochs, dt_labeled, dt_weak, dt_strong, dt_test):
    student, _, _ = load_checkpoint(checkpoint_path=checkpoint_path, optimizer=None, device=device)
    teacher = RobustEMA(student)
    optimizer = torch.optim.SGD(student.parameters(), lr=1e-3, momentum=0.9)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=4, threshold=1e-3, min_lr=1e-5)
    early_stopper = EarlyStopper(min_delta=1e-3)
    cfg_metrics = Metrics(num_classes=21)
    history = {}
    for key in METRIC_SUPERVISED:
        history[f"{key}_supervised"] = []
    for key in METRICS_UNSUPERVISED:
        history[f"{key}_unsupervised"] = []
    history["total"] = []
    history_val = {}
    history_val["validation_loss"] = []

    for epoch in range(epochs):
        print(f"\n==================== Epoch {epoch+1}/{epochs} ====================\n")
        train_history = train_semi_supervised_one_epoch(teacher, student, optimizer, dt_labeled, dt_weak, dt_strong)
        validation_history = validate_semi_supervised(student, dt_test, device, cfg_metrics)
        lr_scheduler.step(validation_history["validation_loss"])
        early_stopper.step(validation_history["validation_loss"])
        for key, val in train_history.items():
            history[key].append(val)

        
        for key, val in validation_history.items():
            if history_val.get(key, None) is None:
                history_val[key] = [val]
            else:
                history_val[key].append(val)
        
        plot_losses(history, METRIC_SUPERVISED, METRICS_UNSUPERVISED, save_dir="results")
        plot_validation_results(history_val, VALIDATION_METRICS,  save_dir="results_val")
        if early_stopper.should_stop:
            print("\nEARLY STOPPING TRIGGERED — Training stopped.\n")
            break


# pipeline_burn_in(50, dt_train_labeled, device, 3)
checkpoint_path="checkpoints/checkpoint_epoch_42.pth"
run_semi_supervised_pipeline(checkpoint_path, 50, dt_train_labeled, dt_train_unlabeled_weakaug, dt_train_unlabeled_strongaug, dt_test)
