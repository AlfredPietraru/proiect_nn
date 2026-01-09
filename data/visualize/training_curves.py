from __future__ import annotations

import os
from typing import Dict, List

from loguru import logger
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class TrainingCurveSupervised:
    def __init__(self, metrics: List[str]) -> None:
        self.metrics = metrics
        self.history = {k: [] for k in metrics}

    def add(self, epoch_history: Dict[str, float]) -> None:
        for k in self.metrics:
            self.history[k].append(float(epoch_history.get(k, 0.0)))

    def plot_total(
        self,
        epoch_history: Dict[str, float],
        save_dir: str,
        filename: str = "train_loss_sup.png",
        show: bool = False,
    ) -> None:
        self.add(epoch_history)

        if "total" not in self.history:
            raise KeyError("Expects metric 'total' in metrics list.")

        os.makedirs(save_dir, exist_ok=True)
        out_path = os.path.join(save_dir, filename)

        epochs = range(1, len(self.history["total"]) + 1)

        plt.figure(figsize=(8, 5))
        plt.plot(list(epochs), self.history["total"], label="Train total", linewidth=2)

        plt.title("Supervised Training Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.tight_layout()

        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        logger.info(f"Plot saved to: {out_path}")

        if show:
            plt.show()
        plt.close()


class TrainingCurvesSemiSupervised:
    def __init__(
        self,
        metrics_supervised: List[str],
        metrics_unsupervised: List[str],
        eval_metrics: List[str],
    ) -> None:
        self.metrics_supervised = metrics_supervised
        self.metrics_unsupervised = metrics_unsupervised
        self.eval_metrics = eval_metrics

        self.history_sup = {k: [] for k in metrics_supervised}
        self.history_unsup = {k: [] for k in metrics_unsupervised}

        self.history_total_loss: List[float] = []
        self.history_eval_loss: List[float] = []

        self.history_eval = {k: [] for k in eval_metrics}

    def add(
        self,
        sup: Dict[str, float],
        unsup: Dict[str, float],
        total_loss: float,
        eval_hist: Dict[str, float],
        val_loss: float,
    ) -> None:
        for k in self.metrics_supervised:
            self.history_sup[k].append(float(sup.get(k, 0.0)))

        for k in self.metrics_unsupervised:
            self.history_unsup[k].append(float(unsup.get(k, 0.0)))

        self.history_total_loss.append(float(total_loss))
        self.history_eval_loss.append(float(val_loss))

        for k in self.eval_metrics:
            self.history_eval[k].append(float(eval_hist.get(k, 0.0)))

    def plot_losses(
        self,
        save_dir: str,
        filename: str = "train_loss_semi.png",
        show: bool = False, plot_components: bool = False,
    ) -> None:
        os.makedirs(save_dir, exist_ok=True)
        out_path = os.path.join(save_dir, filename)

        epochs = range(1, len(self.history_total_loss) + 1)

        plt.figure(figsize=(10, 6))

        if plot_components:
            for k in self.metrics_supervised:
                plt.plot(list(epochs), self.history_sup[k], label=f"{k} (sup)", linewidth=2)

            for k in self.metrics_unsupervised:
                plt.plot(list(epochs), self.history_unsup[k], label=f"{k} (unsup)", linewidth=2)

        plt.plot(list(epochs), self.history_total_loss, label="Train total", linewidth=2)
        plt.plot(list(epochs), self.history_eval_loss, label="Val loss", linewidth=2)

        plt.title("Semi-Supervised Training Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.tight_layout()

        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        logger.info(f"Plot saved to: {out_path}")

        if show:
            plt.show()
        plt.close()

    def plot_eval_metrics(
        self,
        save_dir: str,
        filename: str = "eval_metrics_semi.png",
        show: bool = False,
    ) -> None:
        os.makedirs(save_dir, exist_ok=True)
        out_path = os.path.join(save_dir, filename)

        epochs = range(1, len(self.history_eval_loss) + 1)

        plt.figure(figsize=(10, 6))
        for k in self.eval_metrics:
            plt.plot(list(epochs), self.history_eval[k], label=k, linewidth=2)

        plt.title("Validation Metrics")
        plt.xlabel("Epoch")
        plt.ylabel("Metric")
        plt.legend()
        plt.tight_layout()

        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        logger.info(f"Plot saved to: {out_path}")

        if show:
            plt.show()
        plt.close()
