from __future__ import annotations

from typing import Tuple, Dict

import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2


def normalize(image: np.ndarray, **kwargs) -> np.ndarray:
    return image.astype(np.float32) / 255.0


def build_detection_transforms(size: Tuple[int, int]) -> Dict[str, A.Compose]:
    h, w = size
    # Pascal VOC format for bounding boxes: (xmin, ymin, xmax, ymax)
    # Bounding box parameters x1,y1,x2,y2, this format will work for all the datasets
    # all are applied on same bbox format
    # bbox_params = A.BboxParams(format="pascal_voc", label_fields=["labels"])
    bbox_params = A.BboxParams(format="pascal_voc")

    weak = A.Compose([A.Resize(h, w), A.Lambda(image=normalize), ToTensorV2()], bbox_params=bbox_params)
    strong = A.Compose([
        A.Resize(h, w),
        A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=0.8),
        A.GaussianBlur(blur_limit=(3, 7), sigma_limit=(0.1, 2.0), p=0.5),
        A.CoarseDropout(num_holes_range=(3, 3), hole_height_range=(0.05, 0.1), hole_width_range=(0.05, 0.1), p=0.5),
        A.Lambda(image=normalize),
        ToTensorV2(),
    ], bbox_params=bbox_params)

    test = A.Compose([A.Resize(h, w), A.Lambda(image=normalize), ToTensorV2()], bbox_params=bbox_params)

    return {"weak": weak, "strong": strong, "test": test}
