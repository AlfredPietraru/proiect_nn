import torch
from voc import get_dataloaders
from model_factory import get_model
from ema import RobustEMA
import os
from tqdm import tqdm
from box_ops import BoxList
from metrics import DetectionMetrics
from torch.utils.data import DataLoader
from config_params import Metrics
from torchvision.ops import batched_nms
from main_utils import (
    save_checkpoint, load_checkpoint, EarlyStopper, set_seed, HarryPlotter, HarryPlotterSemiSupervised
)

set_seed(42)
SIZE = (128, 128)
CONFIDENCE_THRESHOLD = 0.7
BATCH_SIZE = 2
CHECKPOINT_DIR = "./checkpoints"
METRIC_BURN_IN = ["loss_classifier", "loss_box_reg", "loss_objectness", "loss_rpn_box_reg", "total"]
METRIC_SUPERVISED = ["loss_classifier", "loss_box_reg", "loss_objectness", "loss_rpn_box_reg"]
METRICS_UNSUPERVISED = ["loss_classifier", "loss_objectness"]
VALIDATION_METRICS = ["mAP_50", "mAP_5095", "precision", "recall", "f1"]
LAMBDA_UNSUPERVISED = 2.0
NMS_IOU = 0.5
ITERATION_TO_STOP_AT = 3000
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def train_burn_in(model, optimizer, data, device):
    model.train()
    train_batches = 0
    history = {key : 0.0 for key in METRIC_BURN_IN}

    for images, targets in tqdm(data["burn_in"], desc="Training"):
        # if train_batches == 5: break
        images = [img.to(device) for img in images]
        for target in targets:
            target["boxes"] = target["boxes"].to(device)
            target["labels"] = target["labels"].to(device)
        
        optimizer.zero_grad()
        _, loss_dict = model(images, targets)
        loss = sum(loss_dict.values())
        loss.backward()
        optimizer.step()
        for k, v in loss_dict.items():
            history[k] += v.item()

        history["total"] += loss.item()
        train_batches += 1
    for key in history:
        history[key] = history[key] / train_batches
    return history


def pipeline_burn_in(epochs, data, device, checkpoint_every):
    model = get_model(device=device)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=2, threshold=1e-3, min_lr=1e-6)
    plotter = HarryPlotter(metrics_found=METRIC_BURN_IN)

    for epoch in range(epochs):
        print(f"\n==================== Epoch {epoch+1}/{epochs} ====================\n")
        train_history = train_burn_in(model, optimizer, data, device)
        lr_scheduler.step(train_history["total"])
        plotter.plot_losses_burn_in(epoch_history=train_history, save_dir="graphs")
        if (epoch + 1) % checkpoint_every == 0 or (epoch + 1) == epochs:
            checkpoint_path = os.path.join(CHECKPOINT_DIR, f"checkpoint_epoch_{epoch+1}.pth")
            save_checkpoint(model, optimizer, epoch + 1, checkpoint_path)


def generate_pseudo_labels(model : torch.nn.Module, images : list[torch.Tensor], device):
    model.eval()
    with torch.no_grad():
        images = [img.to(device) for img in images]
        outputs, _ = model(images, None)
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
    

def train_semi_supervised_one_epoch(teacher : RobustEMA, student, optimizer, data):
    student.train()
    train_batches = 0
    history_supervised = {key : 0.0 for key in METRIC_SUPERVISED}
    history_unsupervised = {key : 0.0 for key in METRICS_UNSUPERVISED}
    total_loss = 0.0

    for (img_labeled, targets_labeled), (img_weak, _), (img_strong, _) in tqdm(zip(data["burn_in"], data["train_weak"], data["train_strong"]), desc="Training"):
        if train_batches == ITERATION_TO_STOP_AT: break
        weak_targets = generate_pseudo_labels(teacher.ema, img_weak, device)
        
        img_strong = [img.to(device) for img in img_strong]
        for target in weak_targets:
            target["boxes"] = target["boxes"].to(device)
            target["labels"] = target["labels"].to(device)
        _, loss_dict_unsupervised = student(img_strong, weak_targets)

        img_labeled = [img.to(device) for img in img_labeled]
        for target in targets_labeled:
            target["boxes"] = target["boxes"].to(device)
            target["labels"] = target["labels"].to(device)
        _, loss_dict_supervised = student(img_labeled, targets_labeled)

        optimizer.zero_grad()
        loss = sum(loss_dict_supervised.values()) + LAMBDA_UNSUPERVISED * (loss_dict_unsupervised["loss_classifier"] + loss_dict_unsupervised["loss_objectness"])
        loss.backward()
        optimizer.step()

        teacher.update(student)
        for k in history_unsupervised.keys():
            history_unsupervised[k] += loss_dict_unsupervised[k].item()
        for k in history_supervised.keys():
            history_supervised[k] += loss_dict_supervised[k].item()
        total_loss += loss.item()
        train_batches += 1

    for key in history_unsupervised:
        history_unsupervised[key] = history_unsupervised[key] / train_batches
    for key in history_supervised:
        history_supervised[key] = history_supervised[key] / train_batches
    return history_supervised, history_unsupervised,  total_loss / train_batches


# TODO de schimbat aici la validaree student.train() si de sters torch.no_grad()
def validate_semi_supervised(student, dt_test, device, cfg_metrics : Metrics):
    student.train()
    metrics = DetectionMetrics(cfg_metrics)
    metrics.reset()
    validation_loss = 0
    
    
    for idx, (images, targets) in enumerate(tqdm((dt_test), desc="Validation")):
            if idx == ITERATION_TO_STOP_AT: break
            for target in targets:
                target["boxes"] = target["boxes"].to(device)
                target["labels"] = target["labels"].to(device)
            images = [img.to(device) for img in images]
            outputs, loss_dict = student(images, targets)
            validation_loss += sum(loss_dict.values())
            preds_bl = [BoxList(o["boxes"], o["labels"], o.get("scores", None), (images[0].shape[1], images[0].shape[2])) for o in outputs]
            tgts_bl  = [BoxList(t["boxes"], t["labels"], t.get("scores", None), (images[0].shape[1], images[0].shape[2])) for t in targets]
            metrics.update(preds_bl, tgts_bl)     
    return metrics.compute(), validation_loss / max(1, len(dt_test))
    

def run_semi_supervised_pipeline(checkpoint_path, epochs, data : dict[str, DataLoader]):
    student, _, _ = load_checkpoint(checkpoint_path=checkpoint_path, optimizer=None, device=device)
    teacher = RobustEMA(student)
    optimizer = torch.optim.SGD(student.parameters(), lr=1e-3, momentum=0.9)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=2, threshold=1e-3, min_lr=1e-6)
    early_stopper = EarlyStopper(min_delta=1e-3)
    cfg_metrics = Metrics(num_classes=21)
    plotter = HarryPlotterSemiSupervised(metrics_supervised=METRIC_SUPERVISED, metrics_unsupervised=METRICS_UNSUPERVISED, eval_metrics=VALIDATION_METRICS)

    for epoch in range(epochs):
        print(f"\n==================== Epoch {epoch+1}/{epochs} ====================\n")
        train_hist_supervised, train_hist_unsupervised, train_loss = train_semi_supervised_one_epoch(teacher, student, optimizer, data)
        validation_history, validation_loss = validate_semi_supervised(student, data["test"], device, cfg_metrics)
        print(train_loss, validation_loss)
        lr_scheduler.step(validation_loss)
        early_stopper.step(validation_loss)
        plotter.add_data(train_hist_supervised, train_hist_unsupervised, train_loss, validation_history, validation_loss)
        plotter.plot_losses(save_dir="graphs")
        plotter.plot_eval_metrics(save_dir="graphs")

        if validation_history["validation_loss"] == early_stopper.best_loss:
            save_checkpoint(student, optimizer, epoch+1, "semi_supervised_results.pth")
        
        if early_stopper.should_stop:
            print("\nEARLY STOPPING TRIGGERED — Training stopped.\n")
            break


os.makedirs(CHECKPOINT_DIR, exist_ok=True)
data = get_dataloaders(SIZE, BATCH_SIZE, False)
pipeline_burn_in(50, data, device, 5)
checkpoint_path="checkpoints/checkpoint_epoch_50.pth"
run_semi_supervised_pipeline(checkpoint_path, 50, data)
