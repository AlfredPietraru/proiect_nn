from __future__ import annotations
from __future__ import absolute_import

from .config import (
    ClassInfo,
    VOC_CLASSES, VOC_NAME_TO_ID,
    UAVDT_CLASSES, UAVDT_NAME_TO_ID,
    VISDRONE_CLASSES, VISDRONE_NAME_TO_ID,
    AUAIR_CLASSES, AUAIR_NAME_TO_ID)
from .download import (
    download_voc,
    download_uavdt,
    download_visdrone,
    download_auair,
    download_all_datasets)
from .voc import VOCDataset
from .uavdt import UAVDTDataset
from .visdrone import VisDroneDataset
from .auair import AUAIRDataset

__all__ = [
    "ClassInfo",
    "VOC_CLASSES", "VOC_NAME_TO_ID",
    "UAVDT_CLASSES", "UAVDT_NAME_TO_ID",
    "VISDRONE_CLASSES", "VISDRONE_NAME_TO_ID",
    "AUAIR_CLASSES", "AUAIR_NAME_TO_ID",

    "download_voc",
    "download_uavdt",
    "download_visdrone",
    "download_auair",
    "download_all_datasets",

    "VOCDataset",
    "UAVDTDataset",
    "VisDroneDataset",
    "AUAIRDataset"
]
