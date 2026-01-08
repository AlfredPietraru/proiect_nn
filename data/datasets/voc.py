from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any
import xml.etree.ElementTree as ET

import albumentations as A
import cv2
import torch
from torch.utils.data import Dataset

from data.datasets.config import VOC_CLASSES
from data.datasets.download import download_voc
from utils.logger import Logger


def parse_voc_annotation(
    annotation_path: Path,
    class_to_idx: Dict[str, int]
) -> Dict[str, Any]:
    root = ET.parse(annotation_path).getroot()

    boxes: List[List[float]] = []
    labels: List[int] = []

    for obj in root.findall("object"):
        name = (obj.findtext("name") or "").strip()
        if name not in class_to_idx:
            continue

        difficult = (obj.findtext("difficult") or "0").strip()
        if difficult.isdigit() and int(difficult) == 1:
            continue

        bb = obj.find("bndbox")
        if bb is None:
            continue

        try:
            xmin = float((bb.findtext("xmin") or "").strip())
            ymin = float((bb.findtext("ymin") or "").strip())
            xmax = float((bb.findtext("xmax") or "").strip())
            ymax = float((bb.findtext("ymax") or "").strip())
        except ValueError:
            continue

        if xmax <= xmin or ymax <= ymin:
            continue

        boxes.append([xmin, ymin, xmax, ymax])
        labels.append(class_to_idx[name])

    return {"boxes": boxes, "labels": torch.tensor(labels, dtype=torch.int64)}


def find_voc_dirs(root: Path, year: str) -> List[Path]:
    out: List[Path] = []
    canonical = root / "VOCdevkit" / f"VOC{year}"
    if canonical.exists():
        out.append(canonical)

    for p in root.glob(f"**/VOC{year}*"):
        if not p.is_dir() or p == canonical:
            continue
        if (p / "JPEGImages").exists() and (p / "Annotations").exists() and (p / "ImageSets" / "Main").exists():
            out.append(p)

    return out


def read_split_ids(split: str, main_dir: Path) -> List[str]:
    split_file = main_dir / f"{split}.txt"
    if split_file.exists():
        return [x.strip() for x in split_file.read_text(encoding="utf-8").splitlines() if x.strip()]

    if split == "trainval":
        ids: List[str] = []
        for s in ("train", "val"):
            f = main_dir / f"{s}.txt"
            if f.exists():
                ids.extend([x.strip() for x in f.read_text(encoding="utf-8").splitlines() if x.strip()])
        return ids

    return []


def build_voc_pairs(
    root: Path, 
    years: Tuple[str, ...], 
    split: str
) -> Tuple[List[Path], List[Path]]:
    images: List[Path] = []
    annots: List[Path] = []
    seen: Set[Tuple[str, str]] = set()

    for year in years:
        for voc_dir in find_voc_dirs(root, year):
            main_dir = voc_dir / "ImageSets" / "Main"
            if not main_dir.exists():
                continue

            for img_id in read_split_ids(split, main_dir):
                key = (year, img_id)
                if key in seen:
                    continue
                seen.add(key)

                img = voc_dir / "JPEGImages" / f"{img_id}.jpg"
                ann = voc_dir / "Annotations" / f"{img_id}.xml"
                if img.exists() and ann.exists():
                    images.append(img)
                    annots.append(ann)

    return images, annots


def make_target(boxes: List[List[float]], labels: List[int] | torch.Tensor) -> Dict[str, torch.Tensor]:
    boxes_t = torch.tensor(boxes, dtype=torch.float32)

    if isinstance(labels, torch.Tensor):
        labels_t = labels.to(dtype=torch.int64)
    else:
        labels_t = torch.tensor(labels, dtype=torch.int64)

    return {"boxes": boxes_t, "labels": labels_t}


class VOCDataset(Dataset):
    def __init__(
        self,
        details: Logger,
        root: str = "VOC",
        split: str = "train",
        years: Tuple[str, ...] = ("2007", "2012"),
        transform: Optional[A.Compose] = None,
        download: bool = True,
    ) -> None:
        assert split in {"train", "trainval", "val", "test", "train_test"}
        assert all(y in {"2007", "2012"} for y in years)

        self.root = Path(root)
        self.split = split
        self.years = years
        self.transform = transform
        self.details = details

        self.class_to_idx = {VOC_CLASSES[i].name: i for i in VOC_CLASSES}
        self.colors = {i: VOC_CLASSES[i].color for i in VOC_CLASSES}

        if download:
            download_voc(str(self.root), details=self.details, years=self.years, force=False)

        self.images, self.annotations = build_voc_pairs(self.root, self.years, self.split)

        if self.details:
            self.details.info(f"Loaded {len(self.images)} VOC images for split='{self.split}'")

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        img_path = self.images[idx]
        ann_path = self.annotations[idx]

        image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Failed to read image: {img_path}")

        raw = parse_voc_annotation(ann_path, self.class_to_idx)
        boxes: List[List[float]] = raw["boxes"]
        labels: torch.Tensor = raw["labels"]

        if self.transform is not None:
            t = self.transform(image=image, bboxes=boxes, labels=labels.tolist())
            image, boxes = t["image"], t["bboxes"]
            labels = torch.tensor(t["labels"], dtype=torch.int64)

        return image, make_target(boxes, labels)
