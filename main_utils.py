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

    epoch = checkpoint.get('epoch', 0)
    print(f"Resuming from epoch {epoch}")
    return model, optimizer, epoch

class HarryPlotter:
    def __init__(self, metrics_found : list[str]) -> None:
        self.history = {key : [] for key in metrics_found}

    def add_new_data(self, epoch_history):
        for key, val in epoch_history.items():
            self.history[key].append(val)

    def plot_losses_burn_in(self, epoch_history : dict, save_dir=None, filename="train_loss_burn_in.png"):
        sns.set_theme(style="whitegrid")
        plt.figure(figsize=(8, 5))
        self.add_new_data(epoch_history)
        plt.plot(list(range(1, len(self.history["total"]) + 1)), self.history["total"], label="Train total", linewidth=2)

        plt.title("Training Loss Components Over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.tight_layout()
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            out_path = os.path.join(save_dir, filename)
            plt.savefig(out_path, dpi=300, bbox_inches="tight")
            print(f"[INFO] Plot saved to: {out_path}")
        plt.close()

class HarryPlotterSemiSupervised:
    def __init__(self, metrics_supervised: list[str], metrics_unsupervised: list[str], eval_metrics: list[str]):
        self.metrics_supervised = metrics_supervised
        self.metrics_unsupervised = metrics_unsupervised
        self.eval_metrics = eval_metrics

        # Training history
        self.history_supervised = {k: [] for k in metrics_supervised}
        self.history_unsupervised = {k: [] for k in metrics_unsupervised}
        self.history_total_loss = []

        # Evaluation history
        self.history_eval = {k: [] for k in eval_metrics}
        self.history_eval_loss = []

    def add_train_data(self, supervised: dict, unsupervised: dict, total_loss: float):
        for k, v in supervised.items():
            if k in self.history_supervised:
                self.history_supervised[k].append(v)
        for k, v in unsupervised.items():
            if k in self.history_unsupervised:
                self.history_unsupervised[k].append(v)
        self.history_total_loss.append(total_loss)

    def add_eval_data(self, eval_history: dict, validation_loss: float):
        for key, val in eval_history.items():
            if key in self.history_eval:
                self.history_eval[key].append(val)
        self.history_eval_loss.append(validation_loss)

    def add_data(self, supervised : dict, unsupervised : dict, total_loss : float, eval_history : dict, validation_loss : float):
        self.add_train_data(supervised, unsupervised, total_loss)
        self.add_eval_data(eval_history, validation_loss)

    def plot_losses(self, save_dir=None, filename="semi_train_loss.png"):
        sns.set_theme(style="whitegrid")
        plt.figure(figsize=(10, 6))
        epochs = range(1, len(self.history_total_loss) + 1)

        # # Supervised losses
        # for k in self.metrics_supervised:
        #     plt.plot(epochs, self.history_supervised[k], label=f"{k} (sup)", linewidth=2)
        # # Unsupervised losses
        # for k in self.metrics_unsupervised:
        #     plt.plot(epochs, self.history_unsupervised[k], label=f"{k} (unsup)", linewidth=2)
        # Total loss
        plt.plot(epochs, self.history_total_loss, label="Total Loss", linewidth=2, color="black")
        plt.plot(epochs, self.history_eval_loss, label="Validation Loss", linewidth=2, color="black")

        plt.title("Training Loss Components Over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.tight_layout()
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            out_path = os.path.join(save_dir, filename)
            plt.savefig(out_path, dpi=300, bbox_inches="tight")
            print(f"[INFO] Training loss plot saved to: {out_path}")
        plt.close()

    def plot_eval_metrics(self, save_dir=None, filename="semi_eval_metrics.png"):
        sns.set_theme(style="whitegrid")
        plt.figure(figsize=(10, 6))
        epochs = range(1, len(self.history_eval_loss) + 1)
        for key in self.eval_metrics:
            plt.plot(epochs, self.history_eval[key], label=key, linewidth=2)

        plt.title("Validation Metrics Over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Metric Value")
        plt.legend()
        plt.tight_layout()
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            out_path = os.path.join(save_dir, filename)
            plt.savefig(out_path, dpi=300, bbox_inches="tight")
            print(f"[INFO] Evaluation metrics plot saved to: {out_path}")
        plt.close()



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
