import shutil
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path
from PIL import Image
import cv2 
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
import numpy as np

from config_data import VOC_CLASSES, VOC_KAGGLE_DATASETS
from logger import Logger
from typing import Tuple


def parse_voc_annotation(annotation_path: Path, class_to_idx: dict) -> dict:
    """Parse a VOC XML annotation file into a dictionary of tensors."""
    tree = ET.parse(annotation_path)
    root = tree.getroot()

    size = root.find('size')
    if size is None:
        return {}

    width_elem = size.find('width')
    height_elem = size.find('height')
    if width_elem is None or height_elem is None:
        return {}

    width = int(width_elem.text or "0")
    height = int(height_elem.text or "0")
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
        if label not in class_to_idx:
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
        labels.append(class_to_idx[label])

    # boxes_tensor = torch.tensor(boxes, dtype=torch.float32)\
    #     if boxes else torch.zeros((0, 4))
    labels_tensor = torch.tensor(labels, dtype=torch.int64)\
        if labels else torch.zeros(0, dtype=torch.int64)

    return {
        "boxes": boxes,
        'labels': labels_tensor,
        # 'image_id': torch.tensor([int(annotation_path.stem)]),
        # 'area': (boxes_tensor[:, 3] - boxes_tensor[:, 1]) * (boxes_tensor[:, 2] - boxes_tensor[:, 0])\
        #     if len(boxes) > 0 else torch.zeros(0),
        # 'iscrowd': torch.zeros(len(boxes), dtype=torch.int64) if boxes else torch.zeros(0, dtype=torch.int64),
        # 'orig_size': torch.tensor([height, width])
    }


class VOCDataset(Dataset):
    def __init__(
        self,
        root: str = "VOC",
        split: str = "train",
        years: list[str] | None = None,
        transform=None,
        download: bool = False,
        details: Logger | None = None,
    ):
        assert split in ["train", "trainval", "val", "test", "train_test"], "Invalid split name"
        assert years is None or all(year in ["2007", "2012"] for year in years), \
            "Years must be '2007' and/or '2012'"

        self.root = Path(root)
        self.split = split
        self.years = years or ["2007", "2012"]
        self.transform = transform
        self.details = details
        # Class mappings
        self.class_to_idx = {VOC_CLASSES[idx].name: idx for idx in VOC_CLASSES}
        self.colors = {idx: VOC_CLASSES[idx].color for idx in VOC_CLASSES}
        # Download dataset if requested
        if download:
            self.download()
        self.images, self.annotations = self.load_data()

    def download(self) -> None:
        """Download Pascal VOC from Kaggle directly to correct path structure."""
        final_devkit = self.root / "VOCdevkit"
        final_devkit.mkdir(parents=True, exist_ok=True)
        
        for year in self.years:
            year = str(year)
            target_dir = final_devkit / f"VOC{year}"
            # Skip if already exists
            if target_dir.exists() and (target_dir / "JPEGImages").exists():
                if self.details:
                    self.details.info(f"VOC{year} exists, skipping download")
                continue
            if year not in VOC_KAGGLE_DATASETS:
                raise RuntimeError(f"No Kaggle dataset for VOC{year}")

            kaggle_slug = VOC_KAGGLE_DATASETS[year]
            if self.details:
                self.details.info(f"Downloading VOC{year}: {kaggle_slug}")

            # Download directly to target directory
            cmd = [
                "kaggle", "datasets", "download",
                "-d", kaggle_slug,
                "-p", str(target_dir),
                "--unzip"
            ]

            try:
                subprocess.run(cmd, check=True)
            except FileNotFoundError:
                raise RuntimeError("Kaggle CLI not found. Install with: pip install kaggle")
            except subprocess.CalledProcessError as e:
                raise RuntimeError(f"Kaggle download failed: {kaggle_slug} (code: {e.returncode})")

            if self.details:
                self.details.info(f"VOC{year} downloaded to: {target_dir}")

    def load_data(self) -> tuple[list[Path], list[Path]]:
        """Load image and annotation file paths for the specified split and years."""
        images: list[Path] = []
        annotations: list[Path] = []
        seen_ids: set[tuple[str, str]] = set()

        for year in self.years:
            year_str = str(year)
            
            # Look for VOC directories
            voc_dirs: list[Path] = []
            canonical = self.root / "VOCdevkit" / f"VOC{year_str}"
            if canonical.exists():
                voc_dirs.append(canonical)
            
            # Also check for alternative locations
            for candidate in self.root.glob(f"**/VOC{year_str}*"):
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
                split_file = main_dir / f"{self.split}.txt"
                
                if split_file.exists():
                    with open(split_file, "r", encoding="utf-8") as f:
                        image_ids = [line.strip() for line in f if line.strip()]
                elif self.split == "trainval":
                    # Fallback: combine train + val
                    for split_name in ["train", "val"]:
                        fallback_file = main_dir / f"{split_name}.txt"
                        if fallback_file.exists():
                            with open(fallback_file, "r", encoding="utf-8") as f:
                                image_ids.extend([line.strip() for line in f if line.strip()])

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

        if self.details:
            self.details.info(f"Loaded {len(images)} images for split='{self.split}'")
        return images, annotations

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict]:
        image = cv2.imread(filename=str(self.images[idx]))
        target = parse_voc_annotation(self.annotations[idx], self.class_to_idx)
        if self.transform:
            transformed = self.transform(image=image, bboxes=target["boxes"])
            image, target["boxes"] = transformed["image"], transformed["bboxes"]
        return image, target


def get_dataloaders(size : Tuple[int, int], batch_size : int, download : bool) -> dict[str, DataLoader]:
    def collate_fn(batch):
        for item in batch:
            item[1]["boxes"] = torch.tensor(item[1]["boxes"], dtype=torch.float32)
        return [item[0] for item in batch], [item[1] for item in batch]
    
    def scale_to_01(image, **kwargs):
        return image.astype('float32') / 255.0

    weak_augmentations = A.Compose([
        A.Resize(size[0], size[1]),         
        A.Lambda(image=scale_to_01), 
        ToTensorV2(),
    ], bbox_params=A.BboxParams(format='pascal_voc'))

    strong_augmentations = A.Compose(
        [
            A.Resize(size[0], size[1]),
            A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=0.8),
            A.GaussianBlur(blur_limit=(3, 7), sigma_limit=(0.1, 2.0), p=0.5),
            A.CoarseDropout(num_holes_range=(3, 3), hole_height_range=(0.05, 0.1),
                             hole_width_range=(0.05, 0.1), p=0.5),
            A.Lambda(image=scale_to_01), 
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(format='pascal_voc')
    )

    test_transforms = A.Compose([
        A.Resize(size[0], size[1]),
        A.Lambda(image=scale_to_01), 
        ToTensorV2(), 
    ], bbox_params=A.BboxParams(format='pascal_voc'))


    ds_train_labeled = VOCDataset(root="VOC", split="trainval", years=["2007"], transform=weak_augmentations, download=download details=None)
    ds_train_unlabeled_weakaug = VOCDataset(root="VOC", split="trainval", years=["2012"], transform=weak_augmentations,  download=download, details=None)
    ds_train_unlabeled_strongaug = VOCDataset(root="VOC", split="trainval", years=["2012"], transform=strong_augmentations, details=None)
    ds_test = VOCDataset(root="VOC", split="test", years=["2007"], transform=test_transforms, details=None)

    dt_train_labeled = DataLoader(ds_train_labeled, batch_size=batch_size, shuffle=True, collate_fn=collate_fn) 
    dt_train_unlabeled_weakaug =  DataLoader(ds_train_unlabeled_weakaug, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    dt_train_unlabeled_strongaug = DataLoader(ds_train_unlabeled_strongaug, batch_size=batch_size, shuffle=True,collate_fn=collate_fn)
    dt_test = DataLoader(ds_test, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    return {
        "burn_in":    dt_train_labeled,
        "train_weak":  dt_train_unlabeled_weakaug,
        "train_strong": dt_train_unlabeled_strongaug,
        "test":        dt_test
    }