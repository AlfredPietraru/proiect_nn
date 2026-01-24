from __future__ import annotations

from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class ClassInfo:
    name: str
    color: str


# ----------------------------
# VOC (Pascal VOC 20 classes)
# ids: 0..19
# ----------------------------
VOC_CLASSES: dict[int, ClassInfo] = {
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
# Mapping from class name to class id for VOC
VOC_NAME_TO_ID: dict[str, int] = {v.name: k for k, v in VOC_CLASSES.items()}


# ----------------------------
# UAVDT (4 classes from your stats)
# ids: 1..4
# ----------------------------
UAVDT_CLASSES: dict[int, ClassInfo] = {
    1: ClassInfo("car", "#06D6A0"),
    2: ClassInfo("vehicle", "#118AB2"),
    3: ClassInfo("truck", "#FFD166"),
    4: ClassInfo("bus", "#EF476F"),
}
# Mapping from class name to class id for UAVDT
UAVDT_NAME_TO_ID: dict[str, int] = {v.name: k for k, v in UAVDT_CLASSES.items()}


# ----------------------------
# VisDrone-DET (standard 10 classes)
# ids: 1..10
# ----------------------------
VISDRONE_CLASSES: dict[int, ClassInfo] = {
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
VISDRONE_NAME_TO_ID: dict[str, int] = {v.name: k for k, v in VISDRONE_CLASSES.items()}


# ----------------------------
# AU-AIR (common road users / traffic)
# ids: 1..8
# ----------------------------
AUAIR_CLASSES: dict[int, ClassInfo] = {
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
AUAIR_NAME_TO_ID: dict[str, int] = {v.name: k for k, v in AUAIR_CLASSES.items()}


def make_target(boxes: list[list[float]], labels: list[int] | torch.Tensor) -> dict[str, torch.Tensor]:
    if boxes is None:
        boxes = []
    if labels is None:
        labels = []

    boxes_t = torch.as_tensor(boxes, dtype=torch.float32)
    labels_t = torch.as_tensor(labels, dtype=torch.int64)

    if boxes_t.numel() == 0:
        boxes_t = boxes_t.reshape(0, 4)
    if labels_t.numel() == 0:
        labels_t = labels_t.reshape(0)

    return {"boxes": boxes_t, "labels": labels_t}
