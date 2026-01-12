from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import random
from pathlib import Path
from xml.etree import ElementTree as ET

import albumentations as A
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
import numpy as np
import random

from data.datasets.config import VOC_CLASSES, VOC_NAME_TO_ID, make_target
from data.datasets.download import download_voc
from utils.logger import Logger
from typing import Tuple


def find_voc_pairs(root: Path, split: str, years: List[int], percentage: float = 1.0) -> Tuple[List[Path], List[Path]]:
    """Load image and annotation file paths for the specified split and years."""
    images: list[Path] = []
    annotations: list[Path] = []
    seen_ids: set[tuple[str, str]] = set()

    for year in years:
        year_str = str(year)
        
        # Look for VOC directories
        voc_dirs: list[Path] = []
        canonical = root / "VOCdevkit" / f"VOC{year_str}"
        if canonical.exists():
            voc_dirs.append(canonical)
        
        # Also check for alternative locations
        for candidate in root.glob(f"**/VOC{year_str}*"):
            if candidate.is_dir() and candidate != canonical:
                if (candidate / "JPEGImages").exists() and \
                (candidate / "Annotations").exists() and \
                (candidate / "ImageSets" / "Main").exists():
                    voc_dirs.append(candidate)

        if not voc_dirs:
            continue

        # Process each VOC directory
        for voc_dir in voc_dirs:
            main_dir = voc_dir / "ImageSets" / "Main"
            if not main_dir.exists():
                continue

            # Get image IDs from split file
            image_ids: list[str] = []
            split_file = main_dir / f"{split}.txt"
            
            # Read image IDs from the split file
            if split_file.exists():
                with open(split_file, "r", encoding="utf-8") as f:
                    image_ids = [line.strip() for line in f if line.strip()]

            # Combine train and val splits if needed
            elif split == "trainval":
                for split_name in ["train", "val"]:
                    fallback_file = main_dir / f"{split_name}.txt"
                    if fallback_file.exists():
                        with open(fallback_file, "r", encoding="utf-8") as f:
                            image_ids.extend([line.strip() for line in f if line.strip()])

            # Fallback to val split for test if needed
            elif split == "test":
                fallback_file = main_dir / "val.txt"
                if fallback_file.exists():
                    with open(fallback_file, "r", encoding="utf-8") as f:
                        image_ids = [line.strip() for line in f if line.strip()]

            if not image_ids:
                continue

            # Add valid images
            for img_id in image_ids:
                key = (year_str, img_id)
                if key in seen_ids:
                    continue
                seen_ids.add(key)

                img_path = voc_dir / "JPEGImages" / f"{img_id}.jpg"
                ann_path = voc_dir / "Annotations" / f"{img_id}.xml"
                
                if img_path.exists() and ann_path.exists():
                    images.append(img_path)
                    annotations.append(ann_path)
                else:
                    print(f"Path {img_path} is invalid.")

    # Shuffle and sample according to a percentage
    n = len(images)
    k = max(1, int(round(n * percentage)))
    indices = list(range(n))
    random.shuffle(indices)

    indices = indices[:k]
    sampled_images = [images[i] for i in indices]
    sampled_annotations = [annotations[i] for i in indices]
    return sampled_images, sampled_annotations


def parse_voc_xml(annotation_path: Path) -> Dict[str, List]:
    """Parse a VOC XML annotation file into a dictionary of tensors."""
    tree = ET.parse(annotation_path)
    root = tree.getroot()

    size = root.find('size')
    if size is None:
        return {"boxes": [], 'labels': []}

    width_elem = size.find('width')
    height_elem = size.find('height')
    if width_elem is None or height_elem is None:
        return {"boxes": [], 'labels': []}

    boxes, labels = [], []

    for obj in root.findall('object'):
        name_elem = obj.find('name')
        if name_elem is None:
            continue

        # Get object label
        label = name_elem.text
        if label is None:
            continue

        # Skip difficult objects for now
        difficult = obj.find('difficult')
        if difficult is not None and difficult.text is not None and int(difficult.text) == 1:
            continue

        # Skip unknown classes
        if label not in VOC_NAME_TO_ID:
            continue

        # Find bounding box coordinates
        bbox = obj.find('bndbox')
        if bbox is None:
            continue

        # Extract coordinates
        xmin_elem = bbox.find('xmin')
        ymin_elem = bbox.find('ymin')
        xmax_elem = bbox.find('xmax')
        ymax_elem = bbox.find('ymax')
        if xmin_elem is None or ymin_elem is None\
            or xmax_elem is None or ymax_elem is None:
            continue

        xmin_text = xmin_elem.text
        ymin_text = ymin_elem.text
        xmax_text = xmax_elem.text
        ymax_text = ymax_elem.text
        if xmin_text is None or ymin_text is None\
            or xmax_text is None or ymax_text is None:
            continue

        # Convert coordinates to float
        xmin = float(xmin_text)
        ymin = float(ymin_text)
        xmax = float(xmax_text)
        ymax = float(ymax_text)
        boxes.append([xmin, ymin, xmax, ymax])
        labels.append(VOC_NAME_TO_ID[label])

    return {"boxes": boxes, 'labels': labels}


class VOCDataset(Dataset):
    def __init__(
        self,
        details: Logger, years: Tuple[str],
        root: str = "VOC", split: str = "train",
        transform: Optional[A.Compose] = None,
        download: bool = True, percentage : float = 1.0
    ) -> None:
        assert split in ["train", "trainval", "val", "test", "train_test"], "Invalid split name"
        assert years is None or all(year in ["2007", "2012"] for year in years), "Years must be '2007' and/or '2012'"

        self.root = Path(root)
        self.split = split
        self.years = years or ["2007", "2012"]
        self.transform = transform
        self.details = details

        self.class_to_idx = {VOC_CLASSES[idx].name: idx for idx in VOC_CLASSES}
        self.colors = {idx: VOC_CLASSES[idx].color for idx in VOC_CLASSES}

        if download:
            download_voc(self.root, details=self.details, years=self.years, force=False)

        self.images, self.annotations = find_voc_pairs(self.root, self.split, [int(y) for y in self.years], percentage)

        if details:
            details.info(
                f"Loaded {len(self.images)} VOC images for "
                f"split='{self.split}' years={self.years}, percentage={percentage} \n"
                f"labels={list(self.class_to_idx.keys())}, from root='{self.root}'")

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict]:
        img_path = self.images[idx]
        ann_path = self.annotations[idx]

        image = np.array(cv2.imread(str(img_path), cv2.IMREAD_COLOR))
        if image is None:
            raise RuntimeError(f"Failed to read image: {img_path}")

        parsed = parse_voc_xml(ann_path)
        boxes = parsed.get("boxes", [])
        labels = parsed.get("labels", [])

        if self.transform:
            t = self.transform(image=image, bboxes=boxes, labels=labels)
            image, boxes, labels = t["image"], t["bboxes"], t["labels"]

        self.details.debug(f"Image {img_path.name}: boxes={len(boxes)}, labels={labels}")
        return image, make_target(boxes, labels)
