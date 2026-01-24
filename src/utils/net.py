from __future__ import annotations

from typing import Any
import os
import torch
import torch.nn as nn
from loguru import logger
from torchviz import make_dot

DEVICE: torch.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def collect_tensors(obj: Any) -> list[torch.Tensor]:
    """Recursively collect all tensors from the given object."""
    if torch.is_tensor(obj):
        return [obj]

    if isinstance(obj, dict):
        return [v for v in obj.values() if torch.is_tensor(v)]

    if isinstance(obj, (list, tuple)):
        out: list[torch.Tensor] = []
        for item in obj:
            out.extend(collect_tensors(item))
        return out

    return []


def output_to_viz_tensor(output: Any) -> torch.Tensor:
    """Convert the model output to a single tensor suitable for visualization."""
    tensors = collect_tensors(output)
    if not tensors:
        raise TypeError("Couldn't find any tensor in the model output for visualization.")
    return sum(t.sum() for t in tensors)


@torch.no_grad()
def visualize_model(
    model: nn.Module, *model_args: Any,
    arch: str = "model", experiment: str = "run",
    out_dir: str = "output", **model_kwargs: Any
)  -> Any:
    """Visualize the model computation graph and save to file."""
    model = model.to("cpu").eval()

    output = model(*model_args, **model_kwargs)
    y = output_to_viz_tensor(output)

    dot = make_dot(y, params=dict(model.named_parameters()))
    os.makedirs(out_dir, exist_ok=True)

    name = f"{arch}_{experiment}"
    file_stem = os.path.join(out_dir, name)

    dot.format = "png"
    dot.render(file_stem, cleanup=True)
    logger.info(f"Model graph saved to {file_stem}.png")

    return dot


def save_model(out_dir: str, name: str, net: nn.Module) -> str:
    """Save model state dict to checkpoint directory."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{name}.pth")
    torch.save(net.state_dict(), path)
    return path


def load_model(ckpt_dir: str, name: str, net: nn.Module, device: torch.device | None = None) -> nn.Module:
    """Load model state dict from checkpoint directory into the provided model instance."""
    if net is None:
        raise ValueError("Model instance must be provided to load the state dict.")

    path = os.path.join(ckpt_dir, f"{name}.pth")
    if not os.path.isfile(path):
        raise FileNotFoundError(f'Could not find the model file "{path}".')

    map_location = DEVICE if device is None else device
    state = torch.load(path, map_location=map_location)
    net.load_state_dict(state)
    return net


def save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer | None, epoch: int, path: str) -> None:
    """Save model and optimizer state to checkpoint file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ckpt = {"epoch": int(epoch), "model_state_dict": model.state_dict()}
    if optimizer is not None:
        ckpt["optimizer_state_dict"] = optimizer.state_dict()

    torch.save(ckpt, path)
    logger.info(f"Checkpoint saved at {path}")


def load_checkpoint(
    path: str, model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    device: torch.device | None = None
) -> tuple[nn.Module, torch.optim.Optimizer | None, int]:
    """Load model and optimizer state from checkpoint file."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    epoch = int(ckpt.get("epoch", 0))
    logger.info(f"Loaded checkpoint from {path} (epoch={epoch})")
    return model, optimizer, epoch
