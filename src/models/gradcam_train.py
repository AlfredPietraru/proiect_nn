from __future__ import annotations

from typing import Any
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

EPS: float = 1e-6

class GradCAMPP(nn.Module):
    """
    Grad-CAM++ engine.

    Methods:
    - capture_activations: forward hook that stores target layer activations (N,C,H,W)
    - normalize_cam: normalize CAM to [0, 1] per-sample
    - bbox_from_cam: CAM threshold -> single xyxy box in the CAM coordinate system
    - forward: compute Grad-CAM++ for given class indices and return CAM-derived boxes
    - remove_hooks: detach hook handle
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module, strict_hooks: bool = True) -> None:
        super().__init__()
        self.model, self.target_layer = model, target_layer
        self.strict_hooks = bool(strict_hooks)

        self.activations: Tensor | None = None
        self.hook = target_layer.register_forward_hook(self.capture_activations)

    def capture_activations(self, module: nn.Module, inputs: tuple[Any, ...], output: Any) -> None:
        """ Forward hook that captures activations from the target layer."""
        if self.strict_hooks and (not isinstance(output, Tensor) or output.ndim != 4):
            raise ValueError("GradCAMPP target layer must output a 4D NCHW tensor.")
        self.activations = output

    @staticmethod
    def normalize_cam(cam: Tensor) -> Tensor:
        """Normalize CAM to [0, 1] per-sample."""
        cam = cam - cam.amin(dim=(-2, -1), keepdim=True)
        cam = cam / (cam.amax(dim=(-2, -1), keepdim=True) + EPS)
        return cam

    @staticmethod
    def bbox_from_cam(cam01: Tensor, thr: float) -> tuple[Tensor, Tensor]:
        """Extract single xyxy box from CAM thresholding."""
        h, w = cam01.shape
        mask = cam01 >= float(thr)

        if not bool(mask.any()):
            return cam01.new_tensor([-1.0, -1.0, -1.0, -1.0]), cam01.new_zeros(())

        ys = torch.arange(h, device=cam01.device)
        xs = torch.arange(w, device=cam01.device)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")

        x1 = xx[mask].min().to(torch.float32)
        y1 = yy[mask].min().to(torch.float32)
        x2 = xx[mask].max().to(torch.float32)
        y2 = yy[mask].max().to(torch.float32)

        score = cam01[mask].mean().to(torch.float32)
        return torch.stack([x1, y1, x2, y2]), score

    def forward(
        self, x: Tensor, class_idx: Tensor | None = None,
        topk: int = 1, thr: float = 0.35, 
        use_gradients: bool = True, detach_outputs: bool = True
    ) -> tuple[
        Tensor,
        Tensor | None, Tensor | None,
        Tensor | None, Tensor | None
    ]:
        """Compute Grad-CAM++ and return CAM-derived boxes."""
        if x.ndim != 4:
            raise ValueError("Input must be NCHW.")
        if not use_gradients:
            raise RuntimeError("Grad-CAM++ needs gradients.")

        self.activations = None

        n, _, h, w = x.shape
        k = int(max(1, topk))

        x_cam = x if x.requires_grad else x.detach().requires_grad_(True)
        logits = self.model(x_cam)
        a = self.activations
        if a is None:
            raise RuntimeError("No activations captured. Check target layer hook.")

        if class_idx is None:
            labels = torch.topk(logits, k=k, dim=1).indices
        else:
            ci = class_idx.to(x.device, torch.long)
            labels = ci.view(n, 1).repeat(1, k) if ci.ndim == 1 else ci

        boxes = x.new_full((n, k, 4), -1.0)
        scores = x.new_zeros((n, k), torch.float32)
        valid = torch.zeros((n, k), x.device, torch.bool)

        for _, kk in enumerate(range(k)):
            cls = labels[:, kk]
            y = logits.gather(1, cls.view(n, 1)).sum()

            g = torch.autograd.grad(y, a, retain_graph=True, create_graph=False)[0]
            g2, g3 = g * g, g * g * g
            denom = 2.0 * g2 + (a * g3).sum(dim=(2, 3), keepdim=True) + EPS

            alpha = g2 / denom
            wgt = (alpha * F.relu(g)).sum(dim=(2, 3), keepdim=True)

            cam = (wgt * a).sum(dim=1)
            cam = F.relu(cam)
            cam = self.normalize_cam(cam)
            cam = F.interpolate(cam.unsqueeze(1), size=(h, w), mode="bilinear", align_corners=False).squeeze(1)

            if detach_outputs:
                cam = cam.detach()

            for i, _ in enumerate(range(n)):
                b, s = self.bbox_from_cam(cam[i], thr=thr)
                if b[0] >= 0:
                    boxes[i, kk], scores[i, kk], valid[i, kk] = b, s, True

        if detach_outputs:
            logits = logits.detach()
            boxes, labels, scores, valid = boxes.detach(), labels.detach(), scores.detach(), valid.detach()

        return logits, boxes, labels, scores, valid

    def remove_hooks(self) -> None:
        """Remove the registered forward hook."""
        self.hook.remove()
