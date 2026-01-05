from __future__ import annotations

from typing import Tuple, Dict
from torch.utils.data import DataLoader

from augmentations import build_detection_transforms

from datasets.auair import AUAIRDataset
from datasets.uavdt import UAVDTDataset
from datasets.visdrone import VisDroneDataset
from datasets.voc import VOCDataset

from unbiased import UnlabeledDataset

from models.hyperparams import ExperimentConfig
from utils.logger import Logger


def get_dataloaders_voc(
    root: str,
    details: Logger,
    size: Tuple[int, int], batch_size: int,
    num_workers: int, pin_memory: bool,
    download: bool = True
) -> Dict[str, DataLoader]:
    def collate_labeled(batch):
        images = [item[0] for item in batch]
        targets = [item[1] for item in batch]
        return images, targets

    def collate_unlabeled(batch):
        weak_images = [b[0] for b in batch]
        strong_images = [b[1] for b in batch]
        return weak_images, strong_images

    tfms = build_detection_transforms(size)
    weak_augmentations = tfms["weak"]
    strong_augmentations = tfms["strong"]
    test_transforms = tfms["test"]

    # Labeled dataset (burn-in): VOC 2007 trainval
    ds_train_labeled = VOCDataset(details, root, "trainval", ("2007",), weak_augmentations, download)
    ds_test = VOCDataset(details, root, "test", ("2007",), test_transforms, download)

    # Unlabeled base dataset: VOC 2012 trainval (NO transform here) - for teacher SSL
    ds_train_unlabeled = VOCDataset(details, root, "trainval", ("2012",), None, download)
    ds_train_unlabeled = UnlabeledDataset(ds_train_unlabeled, weak_augmentations, strong_augmentations)

    loader_train_labeled = DataLoader(
        ds_train_labeled, batch_size, 
        shuffle=True, collate_fn=collate_labeled, 
        num_workers=num_workers, pin_memory=pin_memory)
    loader_train_unlabeled = DataLoader(
        ds_train_unlabeled, batch_size, 
        shuffle=True, collate_fn=collate_unlabeled,
        num_workers=num_workers, pin_memory=pin_memory)
    loader_test = DataLoader(
        ds_test, batch_size, 
        shuffle=False, collate_fn=collate_labeled, 
        num_workers=num_workers, pin_memory=pin_memory)

    return {
        "train_burn_in_strong": loader_train_labeled,
        "train_weak": loader_train_unlabeled,
        "test": loader_test,
    }


def get_dataloaders_uavdt(
    root: str,
    details: Logger,
    size: Tuple[int, int], batch_size: int,
    num_workers: int, pin_memory: bool
) -> Dict[str, DataLoader]:
    def collate_labeled(batch):
        images = [item[0] for item in batch]
        targets = [item[1] for item in batch]
        return images, targets

    def collate_unlabeled(batch):
        weak_images = [b[0] for b in batch]
        strong_images = [b[1] for b in batch]
        return weak_images, strong_images

    tfms = build_detection_transforms(size)
    weak_augmentations = tfms["weak"]
    strong_augmentations = tfms["strong"]
    test_transforms = tfms["test"]

    # Labeled dataset (burn-in): UAVDT train
    ds_train_labeled = UAVDTDataset(root=root, split="train", transform=weak_augmentations, details=details)
    ds_test = UAVDTDataset(root=root, split="test", transform=test_transforms, details=details)

    # Unlabeled base dataset: UAVDT train (NO transform here) - for teacher SSL
    ds_train_unlabeled = UAVDTDataset(root=root, split="train", transform=None, details=details)
    ds_train_unlabeled = UnlabeledDataset(ds_train_unlabeled, weak_augmentations, strong_augmentations)

    loader_train_labeled = DataLoader(
        ds_train_labeled, batch_size, 
        shuffle=True, collate_fn=collate_labeled,
        num_workers=num_workers, pin_memory=pin_memory)
    loader_train_unlabeled = DataLoader(
        ds_train_unlabeled, batch_size, 
        shuffle=True, collate_fn=collate_unlabeled,
        num_workers=num_workers, pin_memory=pin_memory)
    loader_test = DataLoader(
        ds_test, batch_size, 
        shuffle=False, collate_fn=collate_labeled,
        num_workers=num_workers, pin_memory=pin_memory)

    return {
        "train_burn_in_strong": loader_train_labeled,
        "train_weak": loader_train_unlabeled,
        "test": loader_test,
    }


def get_dataloaders_auair(
    root: str,
    details: Logger,
    size: Tuple[int, int], batch_size: int,
    num_workers: int, pin_memory: bool
) -> Dict[str, DataLoader]:
    def collate_labeled(batch):
        images = [item[0] for item in batch]
        targets = [item[1] for item in batch]
        return images, targets

    def collate_unlabeled(batch):
        weak_images = [b[0] for b in batch]
        strong_images = [b[1] for b in batch]
        return weak_images, strong_images

    tfms = build_detection_transforms(size)
    weak_augmentations = tfms["weak"]
    strong_augmentations = tfms["strong"]
    test_transforms = tfms["test"]

    # Labeled dataset (burn-in): AU-AIR train
    ds_train_labeled = AUAIRDataset(root=root, split="train", transform=weak_augmentations, details=details)
    # Test dataset: AU-AIR test (or val if your dataset uses val)
    ds_test = AUAIRDataset(root=root, split="test", transform=test_transforms, details=details)

    # Unlabeled base dataset: AU-AIR train (NO transform here) - for teacher SSL
    ds_train_unlabeled = AUAIRDataset(root=root, split="train", transform=None, details=details)
    ds_train_unlabeled = UnlabeledDataset(ds_train_unlabeled, weak_augmentations, strong_augmentations)

    loader_train_labeled = DataLoader(
        ds_train_labeled, batch_size, 
        shuffle=True, collate_fn=collate_labeled,
        num_workers=num_workers, pin_memory=pin_memory)
    loader_train_unlabeled = DataLoader(
        ds_train_unlabeled, batch_size, 
        shuffle=True, collate_fn=collate_unlabeled,
        num_workers=num_workers, pin_memory=pin_memory)
    loader_test = DataLoader(
        ds_test, batch_size, 
        shuffle=False, collate_fn=collate_labeled,
        num_workers=num_workers, pin_memory=pin_memory)

    return {
        "train_burn_in_strong": loader_train_labeled,
        "train_weak": loader_train_unlabeled,
        "test": loader_test,
    }


def get_dataloaders_visdrone(
    root: str,
    details: Logger,
    size: Tuple[int, int], batch_size: int,
    num_workers: int, pin_memory: bool
) -> Dict[str, DataLoader]:
    def collate_labeled(batch):
        images = [item[0] for item in batch]
        targets = [item[1] for item in batch]
        return images, targets

    def collate_unlabeled(batch):
        weak_images = [b[0] for b in batch]
        strong_images = [b[1] for b in batch]
        return weak_images, strong_images

    tfms = build_detection_transforms(size)
    weak_augmentations = tfms["weak"]
    strong_augmentations = tfms["strong"]
    test_transforms = tfms["test"]

    # Labeled dataset (burn-in): VisDrone train
    ds_train_labeled = VisDroneDataset(root=root, split="train", transform=weak_augmentations, details=details)
    # Test dataset: VisDrone test (some setups use val for evaluation)
    ds_test = VisDroneDataset(root=root, split="test", transform=test_transforms, details=details)

    # Unlabeled base dataset: VisDrone train (NO transform here) - for teacher SSL
    ds_train_unlabeled = VisDroneDataset(root=root, split="train", transform=None, details=details)
    ds_train_unlabeled = UnlabeledDataset(ds_train_unlabeled, weak_augmentations, strong_augmentations)

    loader_train_labeled = DataLoader(
        ds_train_labeled, batch_size, 
        shuffle=True, collate_fn=collate_labeled, 
        num_workers=num_workers, pin_memory=pin_memory)
    loader_train_unlabeled = DataLoader(
        ds_train_unlabeled, batch_size, 
        shuffle=True, collate_fn=collate_unlabeled,
        num_workers=num_workers, pin_memory=pin_memory)
    loader_test = DataLoader(
        ds_test, batch_size, 
        shuffle=False, collate_fn=collate_labeled, 
        num_workers=num_workers, pin_memory=pin_memory)

    return {
        "train_burn_in_strong": loader_train_labeled,
        "train_weak": loader_train_unlabeled,
        "test": loader_test,
    }


def build_dataloaders(cfg: ExperimentConfig) -> Dict[str, DataLoader]:
    size = (cfg.data.img_size, cfg.data.img_size)

    ds = cfg.data.dataset.lower()
    if ds == "voc":
        return get_dataloaders_voc(
            details=Logger("Dataloaders"), root=cfg.data.root, 
            download=cfg.data.download, # VOC can be downloaded automatically
            batch_size=cfg.data.batch_size, size=size,
            num_workers=cfg.data.num_workers,
            pin_memory=cfg.data.pin_memory,
        )
    if ds == "visdrone":
        return get_dataloaders_visdrone(
            details=Logger("Dataloaders"), root=cfg.data.root,
            batch_size=cfg.data.batch_size, size=size,
            num_workers=cfg.data.num_workers,
            pin_memory=cfg.data.pin_memory,
        )
    if ds == "uavdt":
        return get_dataloaders_uavdt(
            details=Logger("Dataloaders"), root=cfg.data.root,
            batch_size=cfg.data.batch_size, size=size,
            num_workers=cfg.data.num_workers,
            pin_memory=cfg.data.pin_memory,
        )
    if ds == "auair":
        return get_dataloaders_auair(
            details=Logger("Dataloaders"), root=cfg.data.root,
            batch_size=cfg.data.batch_size, size=size,
            num_workers=cfg.data.num_workers,
            pin_memory=cfg.data.pin_memory,
        )

    raise ValueError(f"Unknown dataset='{cfg.data.dataset}'")
