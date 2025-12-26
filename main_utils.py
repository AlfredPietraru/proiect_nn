import torch
import numpy as np
import random
import os
from model_factory import get_model
import matplotlib.pyplot as plt
import seaborn as sns

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def save_checkpoint(model, optimizer, epoch, path):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(checkpoint, path)
    print(f"Checkpoint saved at {path}")

def load_checkpoint(checkpoint_path, optimizer=None, device='cuda'):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model = get_model(device=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Model weights loaded from {checkpoint_path}")
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Optimizer state loaded from {checkpoint_path}")

    epoch = checkpoint.get('epoch', 0)
    print(f"Resuming from epoch {epoch}")
    return model, optimizer, epoch

def plot_losses(history, metrics_supervised, metrics_unsupervised, save_dir=None, filename="train_loss_plot.png"):
    sns.set_theme(style="whitegrid")
    epochs = range(1, len(history["total"]) + 1)
    plt.figure(figsize=(8, 5))

    for comp in metrics_supervised:
        plt.plot(epochs, history[f"{comp}_supervised"], label=f"Train {comp}_supervised", linewidth=2)
    for comp in metrics_unsupervised:
        plt.plot(epochs, history[f"{comp}_unsupervised"], label=f"Train {comp}_unsupervised", linewidth=2)
        
    plt.plot(epochs, history["total"], label="Train total", linewidth=2)

    plt.title("Training Loss Components Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    # plt.show()

    # Save to disk
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        out_path = os.path.join(save_dir, filename)
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"[INFO] Plot saved to: {out_path}")


def plot_validation_results(history, validation_metrics, save_dir=None):
    sns.set_theme(style="whitegrid")
    epochs = range(1, len(history[list(history.keys())[0]]) + 1)
    plt.figure(figsize=(10, 6))

    for metric in validation_metrics:
        if metric in history:
          plt.plot(epochs, history[metric], label=metric, linewidth=2)  

    plt.title("Validation Loss and Detection Metrics Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.legend()
    plt.tight_layout()
    # plt.show()

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        out_path = os.path.join(save_dir, "validation_accuracy_plot.png")
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"[INFO] Validation plot saved to: {out_path}")

    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history["validation_loss"], label="validation_loss", linewidth=2)  

    plt.title("Validation Loss Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.legend()
    plt.tight_layout()
    # plt.show()

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        out_path = os.path.join(save_dir, "validation_loss_plot.png")
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"[INFO] Validation plot saved to: {out_path}")


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
