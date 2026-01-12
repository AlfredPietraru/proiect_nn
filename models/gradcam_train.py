from __future__ import annotations

from typing import Tuple, Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


EPS = 1e-6


class GradCAMPP(nn.Module):
    def __init__(
        self, 
        model: nn.Module, 
        target_layer: nn.Module, 
        strict_hooks: bool = True
    ) -> None:
        super().__init__()
        self.model = model
        self.target_layer = target_layer
        self.strict_hooks = bool(strict_hooks)

        self.activations: Optional[torch.Tensor] = None
        self.hook_ok = False

        self.hook = target_layer.register_forward_hook(self.capture_activations)

    def capture_activations(self, module: nn.Module, inputs: Tuple[Any, ...], output: Any) -> None:
        if self.strict_hooks and module is not self.target_layer:
            raise RuntimeError("Wrong target layer.")
        if not isinstance(output, torch.Tensor):
            raise RuntimeError("Target output must be tensor.")
        if output.ndim != 4:
            raise RuntimeError("Target output must be NCHW.")
        if self.strict_hooks and len(inputs) > 0 and isinstance(inputs[0], torch.Tensor):
            if inputs[0].shape[0] != output.shape[0]:
                raise RuntimeError("Batch mismatch.")
        self.activations = output
        self.hook_ok = True

    @staticmethod
    def normalize_cam(cam: torch.Tensor) -> torch.Tensor:
        cam = cam - cam.amin(dim=(-2, -1), keepdim=True)
        cam = cam / (cam.amax(dim=(-2, -1), keepdim=True) + EPS)
        return cam

    @staticmethod
    def bbox_from_cam(cam01: torch.Tensor, thr: float) -> Tuple[torch.Tensor, torch.Tensor]:
        H, W = cam01.shape
        mask = cam01 >= float(thr)

        if not bool(mask.any()):
            return cam01.new_tensor([-1.0, -1.0, -1.0, -1.0]), cam01.new_zeros(())

        ys = torch.arange(H, device=cam01.device)
        xs = torch.arange(W, device=cam01.device)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")

        x1 = xx[mask].min().to(torch.float32)
        y1 = yy[mask].min().to(torch.float32)
        x2 = xx[mask].max().to(torch.float32)
        y2 = yy[mask].max().to(torch.float32)

        score = cam01[mask].mean().to(torch.float32)
        return torch.stack([x1, y1, x2, y2]), score

    def forward(
        self,
        x: torch.Tensor,
        class_idx: Optional[torch.Tensor] = None,
        topk: int = 1, thr: float = 0.35,
        use_gradients: bool = True,
        detach_outputs: bool = True
    ) -> Tuple[
        torch.Tensor,            # logits
        Optional[torch.Tensor],  # boxes
        Optional[torch.Tensor],  # labels
        Optional[torch.Tensor],  # scores
        Optional[torch.Tensor]   # valid
    ]:
        if not use_gradients:
            raise RuntimeError("Grad-CAM++ needs gradients.")

        self.activations = None
        self.hook_ok = False

        N, _, H, W = x.shape
        K = int(max(1, topk))

        x_cam = x if x.requires_grad else x.detach().requires_grad_(True)

        logits = self.model(x_cam)
        if logits.ndim != 2:
            raise ValueError("Model output must be (N,C).")

        A = self.activations
        if (not self.hook_ok) or (A is None):
            raise RuntimeError("No activations captured.")

        if class_idx is None:
            labels = torch.topk(logits, k=K, dim=1).indices
        else:
            ci = class_idx.to(device=x.device, dtype=torch.long)
            if ci.ndim == 0:
                labels = ci.view(1, 1).repeat(N, K)
            elif ci.ndim == 1:
                labels = ci.view(N, 1).repeat(1, K)
            else:
                labels = ci
                if labels.shape != (N, K):
                    raise ValueError("Bad class_idx shape.")

        boxes = x.new_full((N, K, 4), -1.0)
        scores = x.new_zeros((N, K), dtype=torch.float32)
        valid = torch.zeros((N, K), device=x.device, dtype=torch.bool)

        for k in range(K):
            cls = labels[:, k]
            y = logits.gather(1, cls.view(N, 1)).sum()

            G = torch.autograd.grad(
                y, A, retain_graph=True, 
                create_graph=False, allow_unused=False
            )[0]
            if self.strict_hooks and G.shape != A.shape:
                raise RuntimeError("Bad gradient shape.")

            G2 = G * G
            G3 = G2 * G
            denom = 2.0 * G2 + (A * G3).sum(dim=(2, 3), keepdim=True) + EPS

            alpha = G2 / denom
            w = (alpha * F.relu(G)).sum(dim=(2, 3), keepdim=True)

            cam = (w * A).sum(dim=1)
            cam = F.relu(cam)
            cam = self.normalize_cam(cam)
            cam = F.interpolate(
                cam.unsqueeze(1), size=(H, W), 
                mode="bilinear", align_corners=False).squeeze(1)

            if detach_outputs:
                cam = cam.detach()

            for i in range(N):
                b, s = self.bbox_from_cam(cam[i], thr=thr)
                if b[0] >= 0:
                    boxes[i, k] = b
                    scores[i, k] = s
                    valid[i, k] = True

        if detach_outputs:
            out = (boxes.detach(), labels.detach(), scores.detach(), valid.detach())
        else:
            out = (boxes, labels, scores, valid)

        return out
        # return (logits.detach() if detach_outputs else logits, *out)
    
    def remove_hooks(self) -> None:
        self.hook.remove()
