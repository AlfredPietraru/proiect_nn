from __future__ import annotations

from typing import List, Dict, Tuple, Union, Optional
from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class AnchorPoints:
    points: torch.Tensor            # (N, 2) cx,cy
    image_size: Tuple[int, int]     # (H, W)
    num_points_per_location: int    # no. points per spatial location (usually 1)

    def to(self, device) -> "AnchorPoints":
        return AnchorPoints(
            points=self.points.to(device),
            image_size=self.image_size,
            num_points_per_location=self.num_points_per_location,
        )


class PointGenerator(nn.Module):
    def __init__(
        self,
        strides: List[int],
        dtype: torch.dtype = torch.float32,
        clip_to_image: bool = False,
        remove_empty: bool = False,
        return_points_class: bool = False,
        offset: float = 0.5,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()

        if not len(strides):
            raise ValueError("Strides list cannot be empty")

        self.dtype = dtype
        self.strides = strides
        self.offset = float(offset)

        self.clip_to_image = bool(clip_to_image)
        self.remove_empty = bool(remove_empty)
        self.return_points_class = bool(return_points_class)
        self.eps = float(eps)

        self.base_points_cpu = self.generate_base_points()

        self.num_levels = len(self.base_points_cpu)
        for i, p in enumerate(self.base_points_cpu):
            self.register_buffer(f"base_points_l{i}", p, persistent=True)

        # Caching base points and strides on (device, dtype) basis to avoid redundant transfers
        self.base_cache: Dict[
            Tuple[torch.device, torch.dtype], 
            List[torch.Tensor]
        ] = {}

    @staticmethod
    def clip_points_to_image(points_xy: torch.Tensor, image_size: Tuple[int, int]) -> torch.Tensor:
        H, W = image_size
        x = points_xy[:, 0].clamp(0, W)
        y = points_xy[:, 1].clamp(0, H)
        return torch.stack((x, y), dim=1)

    def num_points_per_location(self) -> List[int]:
        return [p.shape[0] for p in self.base_points_cpu]

    def get_base_point(self, level: int) -> torch.Tensor:
        return getattr(self, f"base_points_l{level}")

    def generate_base_points(self) -> List[torch.Tensor]:
        base_points: List[torch.Tensor] = []
        for _ in self.strides:
            base_points.append(torch.tensor([[0.0, 0.0]], dtype=self.dtype, device="cpu"))
        return base_points

    def get_base_points(
        self,
        device: torch.device,
        dtype: torch.dtype,
    ) -> List[torch.Tensor]:
        key = (device, dtype)
        cached = self.base_cache.get(key, None)
        if cached is not None:
            return cached

        # Transfer base points to the requested device and dtype
        points: List[torch.Tensor] = []
        for i in range(self.num_levels):
            b = self.get_base_point(i).to(device=device, dtype=dtype)
            points.append(b)

        self.base_cache[key] = points
        return points

    def generate_points(
        self,
        feature_shapes: List[Tuple[int, int]],
        device: torch.device, dtype: torch.dtype
    ) -> List[torch.Tensor]:
        if len(feature_shapes) != self.num_levels:
            raise ValueError("Feature shapes length must match number of levels")

        base_points = self.get_base_points(device=device, dtype=dtype)

        points_per_level: List[torch.Tensor] = []
        for (H, W), stride, base in zip(feature_shapes, self.strides, base_points):
            shifts_x = (torch.arange(W, device=device, dtype=dtype) + self.offset) * stride
            shifts_y = (torch.arange(H, device=device, dtype=dtype) + self.offset) * stride

            shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x, indexing="ij")
            shift_x = shift_x.reshape(-1)
            shift_y = shift_y.reshape(-1)

            # Shape stacking to get all points at all locations for this level
            shifts = torch.stack((shift_x, shift_y), dim=1)      # (HW,2)
            pts = base.reshape(1, 2) + shifts                    # (HW,2)
            points_per_level.append(pts)                         # (N,2)

        return points_per_level

    def forward(
        self,
        feature_shapes: List[Tuple[int, int]],
        image_sizes: List[Tuple[int, int]],
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Union[
        List[List[torch.Tensor]],    # per img: per lvl: Tensor (N,4) -> (cx,cy,stride,stride)
        List[List[AnchorPoints]],    # per img: per lvl: AnchorPoints
    ]:
        if device is None:
            device = self.get_base_point(0).device
        if dtype is None:
            dtype = self.get_base_point(0).dtype

        points_levels = self.generate_points(feature_shapes, device=device, dtype=dtype)
        num_per_loc = self.num_points_per_location()

        out = []

        for img_size in image_sizes:
            per_img: List[Union[torch.Tensor, AnchorPoints]] = []
            for lvl_idx, lvl_points in enumerate(points_levels):
                p = lvl_points

                if self.clip_to_image:
                    p = self.clip_points_to_image(p, img_size)
                if self.remove_empty:
                    p = self.remove_invalid_points(p, img_size, eps=self.eps)

                if self.return_points_class:
                    per_img.append(AnchorPoints(p, img_size, num_per_loc[lvl_idx]))
                else:
                    per_img.append(p)  # (N,2) cx,cy
            out.append(per_img)
        return out
