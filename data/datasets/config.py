from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class ClassInfo:
    name: str
    color: str


# ----------------------------
# VOC (Pascal VOC 20 classes)
# ids: 0..19
# ----------------------------
VOC_CLASSES: Mapping[int, ClassInfo] = {
    0: ClassInfo("aeroplane", "#FF6B6B"),
    1: ClassInfo("bicycle", "#4ECDC4"),
    2: ClassInfo("bird", "#FFD166"),
    3: ClassInfo("boat", "#118AB2"),
    4: ClassInfo("bottle", "#073B4C"),
    5: ClassInfo("bus", "#EF476F"),
    6: ClassInfo("car", "#06D6A0"),
    7: ClassInfo("cat", "#7209B7"),
    8: ClassInfo("chair", "#F8961E"),
    9: ClassInfo("cow", "#83C5BE"),
    10: ClassInfo("diningtable", "#E29578"),
    11: ClassInfo("dog", "#9B5DE5"),
    12: ClassInfo("horse", "#00BBF9"),
    13: ClassInfo("motorbike", "#00F5D4"),
    14: ClassInfo("person", "#FF99C8"),
    15: ClassInfo("pottedplant", "#A7C957"),
    16: ClassInfo("sheep", "#577590"),
    17: ClassInfo("sofa", "#F9844A"),
    18: ClassInfo("train", "#277DA1"),
    19: ClassInfo("tvmonitor", "#43AA8B"),
}

VOC_KAGGLE_DATASETS: dict[str, str] = {
    "2007": "zaraks/pascal-voc-2007",                   # Pascal VOC 2007 - usually used for testing
    "2012": "gopalbhattrai/pascal-voc-2012-dataset",    # Pascal VOC 2012 - usually used for training
}
# Mapping from class name to class id for VOC
VOC_NAME_TO_ID: Mapping[str, int] = {v.name: k for k, v in VOC_CLASSES.items()}


# ----------------------------
# UAVDT (4 classes from your stats)
# ids: 1..4
# ----------------------------
UAVDT_CLASSES: Mapping[int, ClassInfo] = {
    1: ClassInfo("car", "#06D6A0"),
    2: ClassInfo("vehicle", "#118AB2"),
    3: ClassInfo("truck", "#FFD166"),
    4: ClassInfo("bus", "#EF476F"),
}
# Mapping from class name to class id for UAVDT
UAVDT_NAME_TO_ID: Mapping[str, int] = {v.name: k for k, v in UAVDT_CLASSES.items()}


# ----------------------------
# VisDrone-DET (standard 10 classes)
# ids: 1..10
# ----------------------------
VISDRONE_CLASSES: Mapping[int, ClassInfo] = {
    1: ClassInfo("pedestrian", "#FF99C8"),
    2: ClassInfo("people", "#9B5DE5"),
    3: ClassInfo("bicycle", "#4ECDC4"),
    4: ClassInfo("car", "#06D6A0"),
    5: ClassInfo("van", "#83C5BE"),
    6: ClassInfo("truck", "#FFD166"),
    7: ClassInfo("tricycle", "#F8961E"),
    8: ClassInfo("awning-tricycle", "#277DA1"),
    9: ClassInfo("bus", "#EF476F"),
    10: ClassInfo("motor", "#00F5D4"),
}
# Mapping from class name to class id for VisDrone
VISDRONE_NAME_TO_ID: Mapping[str, int] = {v.name: k for k, v in VISDRONE_CLASSES.items()}


# ----------------------------
# AU-AIR (common road users / traffic)
# ids: 1..8
# ----------------------------
AUAIR_CLASSES: Mapping[int, ClassInfo] = {
    1: ClassInfo("car", "#06D6A0"),
    2: ClassInfo("truck", "#FFD166"),
    3: ClassInfo("bus", "#EF476F"),
    4: ClassInfo("motorcycle", "#00F5D4"),
    5: ClassInfo("bicycle", "#4ECDC4"),
    6: ClassInfo("person", "#FF99C8"),
    7: ClassInfo("train", "#277DA1"),
    8: ClassInfo("other", "#118AB2"),
}
# Mapping from class name to class id for AU-AIR
AUAIR_NAME_TO_ID: Mapping[str, int] = {v.name: k for k, v in AUAIR_CLASSES.items()}
