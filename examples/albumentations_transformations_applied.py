import torch
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

img = np.zeros((100, 100, 3), dtype=np.uint8)
img[:] = (255, 0, 0)  # red image


boxes = [
    [10, 10, 30, 30],
    [50, 50, 80, 80],
]
labels = [1, 2]


transform = A.Compose([
    A.Resize(200, 200),         
    A.HorizontalFlip(p=1.0),
], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))

# --- 4. Apply transform ---
transformed = transform(image=img, bboxes=boxes, labels=labels)
transformed_img = transformed['image']
transformed_boxes = transformed['bboxes']
transformed_labels = transformed['labels']

# --- 5. Inspect results ---
print("Original boxes:", boxes)
print("Transformed boxes:", transformed_boxes)
print("Original image shape:", img.shape)
print("Transformed image shape:", transformed_img.shape)
print("Original labels:", labels)
# TODO WILL HAVE TO CONVERT BACK TO INT, THE ALBUMENTATION LIBRARY CONVERTS THEM TO FLOAT
print("Transformed labels:", transformed_labels)
