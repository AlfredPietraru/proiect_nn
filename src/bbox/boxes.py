from __future__ import annotations

from dataclasses import dataclass
import torch
from torch import Tensor
import torch.nn as nn

EPS: float = 1e-6

@dataclass
class AnchorBoxes:
    boxes: Tensor                   # (N, 4) x1,y1,x2,y2
    image_size: tuple[int, int]     # (H, W)
    num_anchors_per_location: int   # no. anchors per spatial location (usually >1)

    def to(self, device) -> AnchorBoxes:
        return AnchorBoxes(
            boxes=self.boxes.to(device),
            image_size=self.image_size,
            num_anchors_per_location=self.num_anchors_per_location)


class AnchorGenerator(nn.Module):
    def __init__(
        self,
        sizes: list[list[float]],
        aspect_ratios: list[list[float]],
        strides: list[int],
        dtype: torch.dtype = torch.float32,
        clip_to_image: bool = False,
        remove_empty: bool = False,
        return_anchors_class: bool = False,
        offset: float = 0.5,
    ) -> None:
        super().__init__()

        if not (len(sizes) == len(aspect_ratios) == len(strides)):
            raise ValueError("Sizes and aspect ratios length must match no. levels in a YOLO model")

        self.dtype = dtype
        self.sizes = sizes
        self.offset = float(offset)
        self.aspect_ratios = aspect_ratios
        self.strides = strides

        self.clip_to_image = bool(clip_to_image)
        self.remove_empty = bool(remove_empty)
        self.return_anchors_class = bool(return_anchors_class)

        self.base_anchors_cpu = self.generate_base_anchors()

        self.num_levels = len(self.base_anchors_cpu)
        for i, b in enumerate(self.base_anchors_cpu):
            self.register_buffer(f"base_anchors_l{i}", b, persistent=True)

        # Caching anchor boxes on (device, dtype) basis to avoid redundant transfers
        self.base_cache: dict[tuple[torch.device, torch.dtype], list[Tensor]] = {}

    @staticmethod
    def clip_boxes_to_image(boxes: Tensor, image_size: tuple[int, int]) -> Tensor:
        H, W = image_size
        x1 = boxes[:, 0].clamp(0, W)
        y1 = boxes[:, 1].clamp(0, H)
        x2 = boxes[:, 2].clamp(0, W)
        y2 = boxes[:, 3].clamp(0, H)
        return torch.stack((x1, y1, x2, y2), dim=1)

    @staticmethod
    def remove_empty_boxes(boxes: Tensor) -> Tensor:
        keep = (boxes[:, 2] - boxes[:, 0] > EPS) & (boxes[:, 3] - boxes[:, 1] > EPS)
        return boxes[keep]

    def num_anchors_per_location(self) -> list[int]:
        return [self.get_base_anchor(i).shape[0] for i in range(self.num_levels)]

    def get_base_anchor(self, level: int) -> Tensor:
        return getattr(self, f"base_anchors_l{level}")

    def generate_base_anchors(self) -> list[Tensor]:
        base_anchors: list[Tensor] = []
        for sizes_per_level, ratios_per_level in zip(self.sizes, self.aspect_ratios):
            anchors = []
            for size in sizes_per_level:
                area = size * size
                for ratio in ratios_per_level:
                    w = (area / ratio) ** 0.5
                    h = w * ratio
                    anchors.append([-0.5 * w, -0.5 * h, 0.5 * w, 0.5 * h])
            base_anchors.append(Tensor(anchors, self.dtype, "cpu"))
        return base_anchors

    def get_base_anchors(self, device: torch.device,  dtype: torch.dtype) -> list[Tensor]:
        key = (device, dtype)
        cached = self.base_cache.get(key)
        if cached is not None:
            return cached

        # Transfer base anchors to the requested device and dtype
        boxes: list[Tensor] = []
        for i in range(self.num_levels):
            b = self.get_base_anchor(i).to(device, dtype)
            boxes.append(b)

        self.base_cache[key] = boxes
        return boxes

    def generate_anchors(
        self,
        feature_shapes: list[tuple[int, int]],
        device: torch.device, dtype: torch.dtype
    ) -> list[Tensor]:
        if len(feature_shapes) != self.num_levels:
            raise ValueError("Feature shapes length must match number of levels in a YOLO model")

        base_anchors = self.get_base_anchors(device, dtype)

        anchors_per_level: list[Tensor] = []
        for (H, W), stride, base in zip(feature_shapes, self.strides, base_anchors):
            shifts_x = (torch.arange(W, device, dtype) + self.offset) * stride
            shifts_y = (torch.arange(H, device, dtype) + self.offset) * stride

            shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x, indexing="ij")
            shift_x = shift_x.reshape(-1)
            shift_y = shift_y.reshape(-1)

            # Shape stacking to get all anchors at all locations for this level
            shifts = torch.stack((shift_x, shift_y, shift_x, shift_y), dim=1)  # (HW,4)
            anchors = base[None, :, :] + shifts[:, None, :]                    # (HW,A,4)
            anchors_per_level.append(anchors.reshape(-1, 4))                   # (HW*A,4)

        return anchors_per_level

    def forward(
        self,
        feature_shapes: list[tuple[int, int]],
        image_sizes: list[tuple[int, int]],
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ) -> list[list[Tensor]] | list[list[AnchorBoxes]]:
        if device is None:
            device = self.get_base_anchor(0).device
        if dtype is None:
            dtype = self.get_base_anchor(0).dtype

        anchors_levels = self.generate_anchors(feature_shapes, device, dtype)
        num_per_loc = self.num_anchors_per_location()

        out = []

        for img_size in image_sizes:
            per_img: list[Tensor] | list[AnchorBoxes] = []
            for lvl_idx, lvl_anchors in enumerate(anchors_levels):
                a = lvl_anchors

                if self.clip_to_image:
                    a = self.clip_boxes_to_image(a, image_size=img_size)
                if self.remove_empty:
                    a = self.remove_empty_boxes(a)

                if self.return_anchors_class:
                    per_img.append(AnchorBoxes(a, img_size, num_per_loc[lvl_idx]))
                else:
                    per_img.append(torch.cat([a], dim=0))  # (N,4) x1,y1,x2,y2
            out.append(per_img)
        return out
