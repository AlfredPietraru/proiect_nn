from __future__ import annotations

from typing import Tuple, Dict
import albumentations as A
from torch.utils.data import Dataset


class UnlabeledDataset(Dataset):
    # Unlabeled dataset wrapper that applies weak and strong augmentations
    # for teacher part of SSL training - returns both augmented versions

    def __init__(
        self,
        base_dataset: Dataset,
        weak_transform: A.Compose,
        strong_transform: A.Compose,
    ) -> None:
        self.base = base_dataset
        self.weak_tf = weak_transform
        self.strong_tf = strong_transform

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> Tuple[
        A.Compose,  # weakly augmented image
        A.Compose,  # strongly augmented image
        Dict[str, Tuple[float, float, float, float]],  # weak target: boxes + labels
        Dict[str, int]  # strong target: boxes + labels
    ]:
        image, target = self.base[idx]

        boxes, labels = [], []

        # In unlabeled dataset, target is usually empty or None
        if isinstance(target, dict) and "boxes" in target and "labels" in target:
            boxes = target["boxes"].tolist() if hasattr(target["boxes"], "tolist") else target["boxes"]
            labels = target["labels"].tolist() if hasattr(target["labels"], "tolist") else target["labels"]

        w = self.weak_tf(image=image, bboxes=boxes, labels=labels)
        s = self.strong_tf(image=image, bboxes=boxes, labels=labels)

        weak_target = {"boxes": w["bboxes"], "labels": w["labels"]}
        strong_target = {"boxes": s["bboxes"], "labels": s["labels"]}
        return w["image"], s["image"], weak_target, strong_target
