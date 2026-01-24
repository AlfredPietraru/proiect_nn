from __future__ import annotations

import random
from pathlib import Path

import albumentations as A
import json
import cv2
import torch
from torch.utils.data import Dataset

from data.datasets.config import AUAIR_CLASSES, AUAIR_NAME_TO_ID, make_target
from data.datasets.download import download_auair
from utils.logger import Logger


def find_auair_pairs(root: Path, split: str, percentage: float = 1.0) -> tuple[list[Path], list[list[dict]]]:
    """Find AU-AIR image and annotation file pairs."""
    ann_file = root / "annotations" / f"{split}.json"
    img_dir = root / "images"
    if not ann_file.exists() or not img_dir.exists():
        return [], []

    images: list[Path] = []
    annots: list[list[dict]] = []

    data = json.loads(ann_file.read_text(encoding="utf-8"))

    id_to_file: dict[int, Path] = {}
    for im in data.get("images", []):
        img_id = int(im["id"])
        id_to_file[img_id] = img_dir / im["file_name"]

    per_image: dict[Path, list[dict]] = {}
    for ann in data.get("annotations", []):
        img_id = int(ann["image_id"])
        img_path = id_to_file.get(img_id)
        if img_path is None or not img_path.exists():
            continue
        per_image.setdefault(img_path, []).append(ann)

    images = list(per_image.keys())
    images.sort()

    # Shuffle and sample according to a percentage
    n = len(images)
    k = max(1, int(round(n * percentage)))
    indices = list(range(n))
    random.shuffle(indices)

    images = [images[i] for i in indices[:k]]
    annots = [per_image[img] for img in images]
    return images, annots


def parse_auair_anns(anns: Path | list[dict]) -> dict[str, list]:
    """Parse AU-AIR annotations from a Path or a list of annotation dictionaries."""
    boxes, labels = [], []

    if isinstance(anns, Path):
        anns_data = json.loads(anns.read_text(encoding="utf-8"))
    else:
        anns_data = anns

    for a in anns_data:
        bbox = a.get("bbox", None)
        cat = a.get("category_id", None)
        if bbox is None or cat is None:
            continue

        if isinstance(cat, str):
            cat = cat.strip()
            if cat.isdigit():
                cls_id = int(cat)
            else:
                cls_id = AUAIR_NAME_TO_ID.get(cat, -1)
        else:
            cls_id = int(cat)

        if cls_id <= 0:
            continue

        x, y, w, h = bbox
        x1, y1, x2, y2 = float(x), float(y), float(x + w), float(y + h)
        if x2 <= x1 or y2 <= y1:
            continue

        boxes.append([x1, y1, x2, y2])
        labels.append(cls_id)

    return {"boxes": boxes, "labels": labels}


class AUAIRDataset(Dataset):
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

        self.class_to_idx = {AUAIR_CLASSES[i].name: i for i in AUAIR_CLASSES}
        self.colors = {k: v.color for k, v in AUAIR_CLASSES.items()}

        if download:
            download_auair(self.root, details=self.details, force=False, quiet=False)

        self.images, self.annotations = find_auair_pairs(self.root, self.split, percentage)

        if self.details:
            details.info(
                f"Loaded {len(self.images)} AU-AIR images for "
                f"split='{self.split}' percentage={percentage} \n"
                f"labels={list(self.class_to_idx.keys())}, from root='{self.root}'")

    def __len__(self) -> int:
        """Return the number of images in the dataset."""
        return len(self.images)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Get an image and its target annotations (boxes and labels)."""
        img_path = self.images[idx]
        ann_path = self.annotations[idx]

        image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Failed to read image: {img_path}")

        parsed = parse_auair_anns(ann_path)
        boxes = parsed.get("boxes", [])
        labels = parsed.get("labels", [])

        if self.transform is not None:
            t = self.transform(image=image, bboxes=boxes, labels=labels)
            image, boxes, labels = t["image"], t["bboxes"], t["labels"]

        self.details.debug(f"Image {img_path.name}: boxes={len(boxes)}, labels={labels}")
        return image, make_target(boxes, labels)
