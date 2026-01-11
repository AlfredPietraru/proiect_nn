from __future__ import annotations

from pathlib import Path
import random
from typing import Dict, List, Optional, Tuple

import albumentations as A
import cv2
import torch
from torch.utils.data import Dataset

from data.datasets.config import UAVDT_CLASSES, UAVDT_NAME_TO_ID, make_target
from data.datasets.download import download_uavdt
from utils.logger import Logger


def find_uavdt_pairs(root: Path, split: str, percentage: float = 1.0) -> Tuple[List[Path], List[Path]]:
    """Find UAVDT image and annotation file pairs."""
    img_dir = root / "images" / split
    ann_dir = root / "labels" / split

    images: List[Path] = []
    annots: List[Path] = []

    if not img_dir.exists() or not ann_dir.exists():
        return images, annots

    for img_path in sorted(img_dir.glob("*.jpg")):
        ann_path = ann_dir / (img_path.stem + ".txt")
        if ann_path.exists():
            images.append(img_path)
            annots.append(ann_path)

    # Shuffle and sample according to a percentage
    n = len(images)
    k = max(1, int(round(n * percentage)))
    indices = list(range(n))
    random.shuffle(indices)

    images = [images[i] for i in indices[:k]]
    annots = [annots[i] for i in indices[:k]]
    return images, annots


def parse_uavdt_txt(anns: Path) -> Dict[str, List]:
    """Parse UAVDT annotation from a text file."""
    boxes, labels = [], []

    anns_data = anns.read_text(encoding="utf-8")

    for line in anns_data.splitlines():
        line = line.strip()
        if not line:
            continue

        parts = line.split()
        if len(parts) < 5:
            continue

        x = float(parts[0])
        y = float(parts[1])
        w = float(parts[2])
        h = float(parts[3])

        cls_raw = parts[4].strip()

        if cls_raw.isdigit():
            cls_id = int(cls_raw)
        else:
            cls_id = UAVDT_NAME_TO_ID.get(cls_raw, -1)

        if cls_id <= 0:
            continue

        x1, y1, x2, y2 = x, y, x + w, y + h
        if x2 <= x1 or y2 <= y1:
            continue

        boxes.append([x1, y1, x2, y2])
        labels.append(cls_id)

    return {"boxes": boxes, "labels": labels}


class UAVDTDataset(Dataset):
    def __init__(
        self,
        details: Logger,
        root: str, split: str = "train",
        transform: Optional[A.Compose] = None,
        download: bool = True, percentage: float = 1.0
    ) -> None:
        assert split in {"train", "val", "test"}

        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.details = details

        self.class_to_idx = {UAVDT_CLASSES[idx].name: idx for idx in UAVDT_CLASSES}
        self.colors = {idx: UAVDT_CLASSES[idx].color for idx in UAVDT_CLASSES}

        if download:
            download_uavdt(self.root, details=self.details, force=False)

        self.images, self.annotations = find_uavdt_pairs(self.root, self.split, percentage)

        if self.details:
            details.info(
                f"Loaded {len(self.images)} UAVDT images for "
                f"split='{self.split}' percentage={percentage} \n"
                f"labels={list(self.class_to_idx.keys())}, from root='{self.root}'")

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        img_path = self.images[idx]
        ann_path = self.annotations[idx]

        image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Failed to read image: {img_path}")

        parsed = parse_uavdt_txt(ann_path)
        boxes = parsed.get("boxes", [])
        labels = parsed.get("labels", [])

        if self.transform is not None:
            t = self.transform(image=image, bboxes=boxes, labels=labels)
            image, boxes, labels = t["image"], t["bboxes"], t["labels"]

        self.details.debug(f"Image {img_path.name}: boxes={len(boxes)}, labels={labels}")
        return image, make_target(boxes, labels)
