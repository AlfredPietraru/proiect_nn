from __future__ import annotations

from typing import Tuple, Dict
from torch.utils.data import DataLoader, Dataset
import albumentations as A


from .datasets import AUAIRDataset, UAVDTDataset, VisDroneDataset, VOCDataset
from .unbiased import UnlabeledDataset
from data.augmentations import build_detection_transforms
from models.hyperparams import ExperimentConfig
from utils.logger import Logger


def collate_labeled(batch):
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    return images, targets


def collate_unlabeled(batch):
    weak_images = [b[0] for b in batch]
    strong_images = [b[1] for b in batch]
    return weak_images, strong_images


def build_ssl_loaders(
    ds_train_labeled: Dataset, ds_train_unlabeled: Dataset, ds_test: Dataset,
    batch_size: int, num_workers: int, pin_memory: bool,
    weak_augmentations: A.Compose, strong_augmentations: A.Compose
) -> Dict[str, DataLoader]:
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
    return {"train_burn_in_strong": loader_train_labeled, "train_weak": loader_train_unlabeled, "test": loader_test}


def get_dataloaders_voc(
    root: str,
    details: Logger,
    size: Tuple[int, int], batch_size: int,
    num_workers: int, pin_memory: bool,
    download: bool = True, percentage: float = 1.0
) -> Dict[str, DataLoader]:
    tfms = build_detection_transforms(size)
    weak_augmentations = tfms["weak"]
    strong_augmentations = tfms["strong"]
    test_transforms = tfms["test"]
    # Labeled dataset (burn-in + unbiased teacher): VOC 2012 train
    ds_train_labeled = VOCDataset(details, ("2012",), root, "train", weak_augmentations, download, percentage)
    # Unlabeled base dataset: VOC 2012 train (NO transform here) - for teacher SSL
    ds_train_unlabeled = VOCDataset(details, ("2012",), root, "trainval", None, download, percentage)
    # Test dataset: VOC 2007 test
    ds_test = VOCDataset(details, ("2007",), root, "test", test_transforms, download, percentage)
    return build_ssl_loaders(
        ds_train_labeled, ds_train_unlabeled, ds_test,
        batch_size, num_workers, pin_memory, weak_augmentations, strong_augmentations)



def get_dataloaders_uavdt(
    root: str,
    details: Logger,
    size: Tuple[int, int], batch_size: int,
    num_workers: int, pin_memory: bool, 
    download: bool = True, percentage: float = 1.0
) -> Dict[str, DataLoader]:
    tfms = build_detection_transforms(size)
    weak_augmentations = tfms["weak"]
    strong_augmentations = tfms["strong"]
    test_transforms = tfms["test"]
    # Labeled dataset (burn-in + unbiased teacher): UAVDT train
    ds_train_labeled = UAVDTDataset(details, root, "train", weak_augmentations, download, percentage)
    # Unlabeled base dataset: UAVDT train (NO transform here) - for teacher SSL
    ds_train_unlabeled = UAVDTDataset(details, root, "train", None, download, percentage)
    # Test dataset: UAVDT test
    ds_test = UAVDTDataset(details, root, "test", test_transforms, download, percentage)
    return build_ssl_loaders(
        ds_train_labeled, ds_train_unlabeled, ds_test,
        batch_size, num_workers, pin_memory, weak_augmentations, strong_augmentations)


def get_dataloaders_auair(
    root: str,
    details: Logger,
    size: Tuple[int, int], batch_size: int,
    num_workers: int, pin_memory: bool,
    download: bool = True, percentage: float = 1.0
) -> Dict[str, DataLoader]:
    tfms = build_detection_transforms(size)
    weak_augmentations = tfms["weak"]
    strong_augmentations = tfms["strong"]
    test_transforms = tfms["test"]
    # Labeled dataset (burn-in + unbiased teacher): AU-AIR train
    ds_train_labeled = AUAIRDataset(details, root, "train", weak_augmentations, download, percentage)
    # Unlabeled base dataset: AU-AIR train (NO transform here) - for teacher SSL
    ds_test = AUAIRDataset(details, root, "test", test_transforms, download, percentage)
    # Unlabeled base dataset: AU-AIR train (NO transform here) - for teacher SSL
    ds_train_unlabeled = AUAIRDataset(details, root, "train", None, download, percentage)
    return build_ssl_loaders(
        ds_train_labeled, ds_train_unlabeled, ds_test,
        batch_size, num_workers, pin_memory, weak_augmentations, strong_augmentations)


def get_dataloaders_visdrone(
    root: str,
    details: Logger,
    size: Tuple[int, int], batch_size: int,
    num_workers: int, pin_memory: bool,
    download: bool = True, percentage: float = 1.0
) -> Dict[str, DataLoader]:
    tfms = build_detection_transforms(size)
    weak_augmentations = tfms["weak"]
    strong_augmentations = tfms["strong"]
    test_transforms = tfms["test"]
    # Labeled dataset (burn-in + unbiased teacher): VisDrone train
    ds_train_labeled = VisDroneDataset(details, root, "train", weak_augmentations, download, percentage)
    # Unlabeled base dataset: VisDrone train (NO transform here) - for teacher SSL
    ds_train_unlabeled = VisDroneDataset(details, root, "train", None, download, percentage)
    # Test dataset: VisDrone test 
    ds_test = VisDroneDataset(details, root, "test", test_transforms, download, percentage)
    return build_ssl_loaders(
        ds_train_labeled, ds_train_unlabeled, ds_test,
        batch_size, num_workers, pin_memory, weak_augmentations, strong_augmentations)


def build_dataloaders(cfg: ExperimentConfig) -> Dict[str, DataLoader]:
    size = (cfg.data.img_size, cfg.data.img_size)
    details = Logger("Dataloaders")

    ds = cfg.data.dataset.lower()
    if ds == "voc":
        return get_dataloaders_voc(
            cfg.data.root, details, size, cfg.data.batch_size,
            cfg.data.num_workers, cfg.data.pin_memory,
            cfg.data.download, cfg.data.percentage)
    if ds == "visdrone":
        return get_dataloaders_visdrone(
            cfg.data.root, details, size, cfg.data.batch_size,
            cfg.data.num_workers, cfg.data.pin_memory)
    if ds == "uavdt":
        return get_dataloaders_uavdt(
            cfg.data.root, details, size, cfg.data.batch_size,
            cfg.data.num_workers, cfg.data.pin_memory)
    if ds == "auair":
        return get_dataloaders_auair(
            cfg.data.root, details, size, cfg.data.batch_size,
            cfg.data.num_workers, cfg.data.pin_memory)

    raise ValueError(f"Unknown dataset='{cfg.data.dataset}'")
