from __future__ import annotations

from typing import Callable
import os
import numpy as np
import torch
from torch import nn
from torch.utils import data
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.figure as fig
from loguru import logger


def plot_dists(
    values: dict[str, np.ndarray],
    xlabel: None | str = None, bins: int = 30, density: bool = False,
    out_dir: str = "output", file_name: str = "distributions.png"
) -> fig.Figure:
    """
    Histogram distribution plotter for multiple sets of values.
    Distribution is meant for the values in each dict entry.
    Dictinoary are for multiple distributions to plot.
    """
    os.makedirs(out_dir, exist_ok=True)

    keys = sorted(values.keys())
    ncols = max(1, len(keys))

    max_cols = 6
    ncols_eff = min(ncols, max_cols)
    nrows_eff = int(np.ceil(ncols / float(ncols_eff)))

    fig_w = float(ncols_eff) * 3.6
    fig_h = float(nrows_eff) * 3.0
    fig, axes = plt.subplots(nrows_eff, ncols_eff, figsize=(fig_w, fig_h), squeeze=False)

    for ax in axes.reshape(-1):
        ax.axis("off")

    for i, key in enumerate(keys):
        r = i // ncols_eff
        c = i % ncols_eff
        ax = axes[r, c]
        ax.axis("on")

        arr = np.asarray(values[key]).reshape(-1)
        if arr.size == 0:
            ax.set_title(f"{key} (empty)", fontsize=10)
            ax.axis("off")
            continue

        lo = np.nanpercentile(arr, 0.5)
        hi = np.nanpercentile(arr, 99.5)
        if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
            lo = float(np.nanmin(arr))
            hi = float(np.nanmax(arr))
            if lo == hi:
                lo -= 1.0
                hi += 1.0

        ax.hist(
            arr, bins=bins,
            density=density,
            histtype="stepfilled",
            alpha=0.65, linewidth=1.0, edgecolor="black")

        mu = float(np.nanmean(arr))
        med = float(np.nanmedian(arr))
        if np.isfinite(mu):
            ax.axvline(mu, linestyle="--", linewidth=1.2, alpha=0.85)
        if np.isfinite(med):
            ax.axvline(med, linestyle=":", linewidth=1.2, alpha=0.85)

        ax.set_xlim(lo, hi)
        ax.set_title(f"{key}  (n={arr.size})", fontsize=10)

        if xlabel is not None:
            ax.set_xlabel(xlabel, fontsize=9)

        ax.tick_params(axis="both", labelsize=8)
        ax.grid(True, which="major", linewidth=0.6, alpha=0.35)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.tight_layout(pad=0.8, w_pad=0.8, h_pad=0.9)

    save_path = os.path.join(out_dir, file_name)
    fig.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.1)
    logger.info(f"Saved plot to {save_path}")
    return fig


def visualize_weight_distribution(model: nn.Module, out_dir: str = "output", file_name: str = "weights.png") -> None:
    """
    Visualize the weight distribution of all trainable parameters in the model,
    excluding biases, and save the plot to the specified output directory.
    """
    weights: dict[str, np.ndarray] = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.endswith(".bias"):
            continue
        weights[name] = p.detach().cpu().view(-1).numpy()

    if not weights:
        logger.warning("No non-bias trainable parameters found.")
        return

    plt.ioff()
    fig = plot_dists(weights, xlabel="Weight value", out_dir=out_dir, file_name=file_name)
    fig.suptitle("Weight distribution", fontsize=14, y=1.03)
    plt.close(fig)


def visualize_gradients(
    model: nn.Module, train_set: data.Dataset,
    batch_size: int = 256, device: None | torch.device = None,
    out_dir: str = "output", file_name: str = "gradients.png"
) -> None:
    """
    Visualize the gradient distribution of all trainable parameters in the model,
    using a single batch from the provided training dataset, and save the plot
    to the specified output directory.
    """
    if train_set is None:
        raise ValueError("Train set must be provided to visualize gradients.")

    model.train()  # we want gradients to flow (detection losses require train mode)
    if device is None:
        device = next(model.parameters()).device
    model.to(device)

    def det_collate(batch) -> tuple[list[torch.Tensor], list[dict]]:
        images, targets = zip(*batch)
        return list(images), list(targets)

    loader = data.DataLoader(train_set, batch_size=batch_size, shuffle=False, collate_fn=det_collate)
    images, targets = next(iter(loader))

    images = [im.to(device) for im in images]
    model.zero_grad(set_to_none=True)
    loss = None

    if isinstance(targets, list) and len(targets) > 0 and isinstance(targets[0], dict):
        targets = [
            {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in t.items()}
            for t in targets]

        out = model(images, targets)

        if isinstance(out, tuple) and len(out) == 2 and isinstance(out[1], dict):
            loss_dict = out[1]
        elif isinstance(out, dict):
            loss_dict = out
        else:
            loss_dict = {}

        if loss_dict:
            loss = torch.stack([v for v in loss_dict.values()]).sum()

    if loss is None:
        try:
            loader_cls = data.DataLoader(train_set, batch_size=batch_size, shuffle=False)
            imgs, labels = next(iter(loader_cls))
            imgs, labels = imgs.to(device), labels.to(device)
            preds = model(imgs)
            loss = F.cross_entropy(preds, labels)
        except Exception as e:
            logger.warning(f"Could not compute gradients: {e}")
            model.zero_grad(set_to_none=True)
            return

    loss.backward()

    grads: dict[str, np.ndarray] = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.grad is None:
            continue
        if name.endswith(".bias"):
            continue
        grads[name] = p.grad.detach().abs().cpu().reshape(-1).numpy()

    model.zero_grad(set_to_none=True)

    if not grads:
        logger.warning("No gradients collected.")
        return

    plt.ioff()
    fig = plot_dists(grads, xlabel="|grad|", out_dir=out_dir, file_name=file_name)
    fig.suptitle("Gradient distribution", fontsize=14, y=1.03)
    plt.close(fig)


def visualize_activations(
    model: nn.Module, train_set: data.Dataset,
    device: None | torch.device = None,
    batch_size: int = 256, max_samples_per_layer: int = 100_000,
    out_dir: str = "output", file_name: str = "activations.png",
    print_variance: bool = False
) -> None:
    """
    Visualize the activation distribution of all layers in the model,
    using a single batch from the provided training dataset, and save the plot
    to the specified output directory.
    """
    if train_set is None:
        logger.warning("No train set provided, skipping activation visualization.")
        return

    model.eval()
    if device is None:
        device = next(model.parameters()).device
    model.to(device)

    def det_collate(batch) -> tuple[list[torch.Tensor], list[dict]]:
        images, targets = zip(*batch)
        return list(images), list(targets)

    loader = data.DataLoader(train_set, batch_size=batch_size, shuffle=False, collate_fn=det_collate)
    images, _ = next(iter(loader))
    images = [im.to(device) for im in images]

    activations: dict[str, np.ndarray] = {}
    hooks = []

    def register_hook(layer_name: str) -> Callable:
        def hook(_, __, output):
            out = output[0] if isinstance(output, tuple) else output
            flat = out.detach().reshape(-1)
            if flat.numel() > max_samples_per_layer:
                idx = torch.randperm(flat.numel(), device=flat.device)[:max_samples_per_layer]
                flat = flat[idx]
            activations[layer_name] = flat.cpu().numpy()
        return hook

    layer_idx = 0
    for module in model.modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            name = f"{module.__class__.__name__} {layer_idx}"
            hooks.append(module.register_forward_hook(register_hook(name)))
            layer_idx += 1

    with torch.no_grad():
        _ = model(images)

    for h in hooks:
        h.remove()

    if not activations:
        logger.warning("No activations collected (no Linear/Conv2d layers found).")
        return

    if print_variance:
        for k in sorted(activations.keys()):
            logger.info(f"{k} variance: {np.var(activations[k])}")

    plt.ioff()
    fig = plot_dists(
        activations, xlabel="Activation value",
        density=True, out_dir=out_dir, file_name=file_name)
    fig.suptitle("Activation distribution", fontsize=14, y=1.03)
    plt.close(fig)
