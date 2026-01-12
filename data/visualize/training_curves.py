from __future__ import annotations

import os
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import seaborn as sns

from data.visualize.visualize_common import save_figure

sns.set_theme(style="whitegrid")


class TrainingCurveSupervised:
    def __init__(self, metrics: List[str]) -> None:
        self.metrics = metrics
        self.history = {k: [] for k in metrics}

    def update(self, epoch_metrics: Dict[str, float]) -> None:
        for k in self.metrics:
            self.history[k].append(float(epoch_metrics.get(k, 0.0)))

    def plot_total(self, save_dir: str, show: bool = False, save_path: Optional[str] = None) -> None:
        os.makedirs(save_dir, exist_ok=True)
        if "total" not in self.history:
            return

        epochs = list(range(1, len(self.history["total"]) + 1))
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.lineplot(x=epochs, y=self.history["total"], ax=ax, label="Train total", linewidth=2)

        ax.set_title("Supervised Training Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()

        fig.tight_layout()
        if save_path is not None:
            save_figure(fig, save_path)
        if show:
            plt.show()


class TrainingCurveSemiSupervised:
    def __init__(self, metrics_supervised: List[str], metrics_unsupervised: List[str]) -> None:
        self.metrics_supervised = metrics_supervised
        self.metrics_unsupervised = metrics_unsupervised

        # Labeled only
        self.history_sup = {k: [] for k in metrics_supervised}
        # Unlabeled + labeled
        self.history_unsup = {k: [] for k in metrics_unsupervised}
        # Overall total loss
        self.history_total_loss: List[float] = []

    def update_supervised(self, metrics: Dict[str, float]) -> None:
        for k in self.metrics_supervised:
            self.history_sup[k].append(float(metrics.get(k, 0.0)))

    def update_unsupervised(self, metrics: Dict[str, float]) -> None:
        for k in self.metrics_unsupervised:
            self.history_unsup[k].append(float(metrics.get(k, 0.0)))

    def update_total_loss(self, total_loss: float) -> None:
        self.history_total_loss.append(float(total_loss))

    def plot_losses(
        self, plot_components: bool,
        save_dir: str, show: bool = False, save_path: Optional[str] = None
    ) -> None:
        os.makedirs(save_dir, exist_ok=True)
        fig, ax = plt.subplots(figsize=(10, 6))

        epochs = list(range(1, len(self.history_total_loss) + 1))
        sns.lineplot(x=epochs, y=self.history_total_loss, ax=ax, label="Total loss", linewidth=2)

        if plot_components:
            for k in self.metrics_supervised:
                if self.history_sup.get(k):
                    sns.lineplot(
                        x=epochs[:len(self.history_sup[k])],
                        y=self.history_sup[k], 
                        ax=ax, label=f"{k} (sup)", linewidth=2)
            for k in self.metrics_unsupervised:
                if self.history_unsup.get(k):
                    sns.lineplot(
                        x=epochs[:len(self.history_unsup[k])], 
                        y=self.history_unsup[k], 
                        ax=ax, label=f"{k} (unsup)", linewidth=2)

        ax.set_title("Training Loss Curves")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()

        fig.tight_layout()
        if save_path is not None:
            save_figure(fig, save_path)
        if show:
            plt.show()

    def plot_eval_metrics(
        self, metrics: Dict[str, List[float]], 
        save_dir: str, show: bool = False, save_path: Optional[str] = None
    ) -> None:
        os.makedirs(save_dir, exist_ok=True)
        fig, ax = plt.subplots(figsize=(10, 6))

        for k, vals in metrics.items():
            if not vals:
                continue
            epochs = list(range(1, len(vals) + 1))
            sns.lineplot(x=epochs, y=vals, ax=ax, label=k, linewidth=2)

        ax.set_title("Evaluation Metrics")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Metric")
        ax.legend()

        fig.tight_layout()
        if save_path is not None:
            save_figure(fig, save_path)
        if show:
            plt.show()
