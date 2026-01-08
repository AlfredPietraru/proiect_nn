from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import albumentations as A
import cv2
import torch
from torch.utils.data import Dataset

from data.datasets.config import UAVDT_NAME_TO_ID, UAVDT_CLASSES
from data.datasets.download import download_uavdt
from utils.logger import Logger


def find_uavdt_pairs(root: Path, split: str) -> Tuple[List[Path], List[Path]]:
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

    return images, annots


def parse_uavdt_txt(ann_path: Path) -> Tuple[List[List[float]], List[int]]:
    boxes: List[List[float]] = []
    labels: List[int] = []

    for line in ann_path.read_text(encoding="utf-8").splitlines():
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

    return boxes, labels


def make_target(boxes: List[List[float]], labels: List[int] | torch.Tensor) -> Dict[str, torch.Tensor]:
    boxes_t = torch.tensor(boxes, dtype=torch.float32)

    if isinstance(labels, torch.Tensor):
        labels_t = labels.to(dtype=torch.int64)
    else:
        labels_t = torch.tensor(labels, dtype=torch.int64)

    return {"boxes": boxes_t, "labels": labels_t}


class UAVDTDataset(Dataset):
    def __init__(
        self,
        details: Logger,
        root: str,
        split: str = "train",
        transform: Optional[A.Compose] = None,
        download: bool = False,
    ) -> None:
        assert split in {"train", "val", "test"}

        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.details = details

        self.colors = {k: v.color for k, v in UAVDT_CLASSES.items()}

        if download:
            download_uavdt(self.root, details=self.details, force=False)

        self.images, self.annotations = find_uavdt_pairs(self.root, self.split)

        if self.details:
            self.details.info(f"Loaded {len(self.images)} UAVDT images for split='{self.split}'")

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        img_path = self.images[idx]
        ann_path = self.annotations[idx]

        image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Failed to read image: {img_path}")

        boxes, labels = parse_uavdt_txt(ann_path)

        if self.transform is not None:
            t = self.transform(image=image, bboxes=boxes, labels=labels)
            image, boxes, labels = t["image"], t["bboxes"], t["labels"]

        return image, make_target(boxes, labels)
