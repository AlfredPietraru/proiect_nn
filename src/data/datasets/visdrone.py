from __future__ import annotations

from pathlib import Path
import random
import albumentations as A
import cv2
import torch
from torch.utils.data import Dataset

from data.datasets.config import VISDRONE_CLASSES, VISDRONE_NAME_TO_ID, make_target
from data.datasets.download import download_visdrone
from utils.logger import Logger


def find_visdrone_pairs(root: Path, split: str, percentage: float = 1.0) -> tuple[list[Path], list[Path]]:
    """Find VisDrone image and annotation file pairs."""
    img_dir = root / split / "images"
    ann_dir = root / split / "annotations"

    images: list[Path] = []
    annots: list[Path] = []

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


def parse_visdrone_txt(ann_path: Path) -> dict[str, list]:
    """Parse VisDrone annotation from a text file."""
    boxes, labels = [], []

    anns_data = ann_path.read_text(encoding="utf-8")

    for line in anns_data.splitlines():
        line = line.strip()
        if not line:
            continue

        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 6:
            continue

        x = float(parts[0])
        y = float(parts[1])
        w = float(parts[2])
        h = float(parts[3])

        cls_raw = parts[4].strip()

        if cls_raw.isdigit():
            cat = int(float(cls_raw))
        else:
            cat = VISDRONE_NAME_TO_ID.get(cls_raw, -1)

        x1, y1, x2, y2 = x, y, x + w, y + h
        if x2 <= x1 or y2 <= y1:
            continue

        boxes.append([x1, y1, x2, y2])
        labels.append(cat)

    return {"boxes": boxes, "labels": labels}


class VisDroneDataset(Dataset):
    def __init__(
        self,
        details: Logger,
        root: str, split: str = "train",
        transform: A.Compose | None = None,
        download: bool = True, percentage: float = 1.0
    ) -> None:
        assert split in {"train", "val", "test"}

        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.details = details

        self.class_to_idx = {VISDRONE_CLASSES[idx].name: idx for idx in VISDRONE_CLASSES}
        self.colors = {idx: VISDRONE_CLASSES[idx].color for idx in VISDRONE_CLASSES}

        if download:
            download_visdrone(self.root, details=self.details, force=False)

        self.images, self.annotations = find_visdrone_pairs(self.root, self.split, percentage)

        if self.details:
            details.info(
                f"Loaded {len(self.images)} VisDrone images for "
                f"split='{self.split}' percentage={percentage} \n"
                f"labels={list(self.class_to_idx.keys())}, from root='{self.root}'")

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        img_path = self.images[idx]
        ann_path = self.annotations[idx]

        image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Failed to read image: {img_path}")

        parsed = parse_visdrone_txt(ann_path)
        boxes = parsed.get("boxes", [])
        labels = parsed.get("labels", [])

        if self.transform is not None:
            t = self.transform(image=image, bboxes=boxes, labels=labels)
            image, boxes, labels = t["image"], t["bboxes"], t["labels"]

        self.details.debug(f"Image {img_path.name}: boxes={len(boxes)}, labels={labels}")
        return image, make_target(boxes, labels)
