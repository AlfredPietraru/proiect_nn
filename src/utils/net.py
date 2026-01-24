from __future__ import annotations

import os
from typing import List, Tuple, Any, Mapping, Optional

import torch
import torch.nn as nn
from loguru import logger
from torchviz import make_dot


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def collect_tensors(obj: Any) -> List[torch.Tensor]:
    if torch.is_tensor(obj):
        return [obj]

    if isinstance(obj, Mapping):
        return [v for v in obj.values() if torch.is_tensor(v)]

    if isinstance(obj, (List, Tuple)):
        out: List[torch.Tensor] = []
        for item in obj:
            out.extend(collect_tensors(item))
        return out

    return []


def output_to_viz_tensor(output: Any) -> torch.Tensor:
    tensors = collect_tensors(output)
    if not tensors:
        raise TypeError("Couldn't find any tensor in the model output for visualization.")
    return sum(t.sum() for t in tensors)


@torch.no_grad()
def visualize_model(
    model: nn.Module,
    *model_args: Any,
    arch: str = "model",
    experiment: str = "run",
    out_dir: str = "output",
    **model_kwargs: Any,
)  -> Any:
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
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{name}.pth")
    torch.save(net.state_dict(), path)
    return path


def load_model(
    ckpt_dir: str, 
    name: str, net: nn.Module, 
    device: Optional[torch.device]
) -> nn.Module:
    if net is None:
        raise ValueError("Model instance must be provided to load the state dict.")

    path = os.path.join(ckpt_dir, f"{name}.pth")
    if not os.path.isfile(path):
        raise FileNotFoundError(f'Could not find the model file "{path}".')

    map_location = DEVICE if device is None else device
    state = torch.load(path, map_location=map_location)
    net.load_state_dict(state)
    return net
