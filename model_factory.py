
import torch
import torch.nn as nn
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models import ResNet50_Weights 
from typing import List, Any

class FasterRCNN_RESNET50_FPN(nn.Module):
    def __init__(self, num_classes=21):
        super().__init__()
        base = fasterrcnn_resnet50_fpn(
            weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        )
        in_features = base.roi_heads.box_predictor.cls_score.in_features
        base.roi_heads.box_predictor = FastRCNNPredictor(
            in_features, num_classes
        )
        self.backbone = base.backbone
        self.rpn = base.rpn
        self.roi_heads = base.roi_heads
        self.transform = base.transform

    def forward(self, images, targets = None):
        images, targets = self.transform(images, targets)  
        features = self.backbone(images.tensors)
        proposals, rpn_losses = self.rpn(images, features, targets)
        detections, roi_losses = self.roi_heads(features, proposals, images.image_sizes, targets)

        if targets is None:
            return detections, None

        losses = {}
        losses.update(rpn_losses)
        losses.update(roi_losses)
        return detections, losses


    
def get_model(device):
    model = FasterRCNN_RESNET50_FPN()
    return model.to(device)

