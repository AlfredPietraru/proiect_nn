from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json

import albumentations as A
import cv2
import torch
from torch.utils.data import Dataset

from config import AUAIR_CLASSES, AUAIR_NAME_TO_ID
from download import download_auair
from ...utils.logger import Logger


def load_auair_index(root: Path, split: str) -> Tuple[List[Path], Dict[str, List[dict]]]:
    ann_file = root / "annotations" / f"{split}.json"
    img_dir = root / "images"

    if not ann_file.exists() or not img_dir.exists():
        return [], {}

    data = json.loads(ann_file.read_text(encoding="utf-8"))

    id_to_file: Dict[int, Path] = {}
    for im in data.get("images", []):
        img_id = int(im["id"])
        file_name = im["file_name"]
        id_to_file[img_id] = img_dir / file_name

    per_image: Dict[str, List[dict]] = {}
    for ann in data.get("annotations", []):
        img_id = int(ann["image_id"])
        img_path = id_to_file.get(img_id)
        if img_path is None:
            continue
        per_image.setdefault(str(img_path), []).append(ann)

    images = [Path(k) for k in per_image.keys() if Path(k).exists()]
    images.sort()
    return images, per_image


def parse_auair_anns(anns: List[dict]) -> Tuple[List[List[float]], List[int]]:
    boxes: List[List[float]] = []
    labels: List[int] = []

    for a in anns:
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

    return boxes, labels


def make_target(boxes: List[List[float]], labels: List[int] | torch.Tensor) -> Dict[str, torch.Tensor]:
    boxes_t = torch.tensor(boxes, dtype=torch.float32)

    if isinstance(labels, torch.Tensor):
        labels_t = labels.to(dtype=torch.int64)
    else:
        labels_t = torch.tensor(labels, dtype=torch.int64)

    return {"boxes": boxes_t, "labels": labels_t}


class AUAIRDataset(Dataset):
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

        self.class_to_idx = {AUAIR_CLASSES[i].name: i for i in AUAIR_CLASSES}
        self.colors = {k: v.color for k, v in AUAIR_CLASSES.items()}

        if download:
            download_auair(self.root, details=self.details, force=False, quiet=False)

        self.images, self.per_image = load_auair_index(self.root, self.split)

        if self.details:
            self.details.info(f"Loaded {len(self.images)} AU-AIR images for split='{self.split}'")

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        img_path = self.images[idx]

        image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Failed to read image: {img_path}")

        anns = self.per_image.get(str(img_path), [])
        boxes, labels = parse_auair_anns(anns)

        if self.transform is not None:
            t = self.transform(image=image, bboxes=boxes, labels=labels)
            image, boxes, labels = t["image"], t["bboxes"], t["labels"]

        return torch.tensor(image), make_target(boxes, labels)
