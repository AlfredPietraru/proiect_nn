from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
from torchvision.models.resnet import resnet50
from torchvision.models.resnet import ResNet50_Weights
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.rpn import RegionProposalNetwork, RPNHead
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.image_list import ImageList
from torchvision.models.detection.roi_heads import fastrcnn_loss
from torchvision.ops import MultiScaleRoIAlign
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_Weights,
    fasterrcnn_resnet50_fpn,
)

from .hyperparams import ExperimentConfig


class FasterRCNNResNet50FPN(nn.Module):
    def __init__(self, cfg : ExperimentConfig) -> None:
        super().__init__()
        # num_classes_with_bg: int,
        # img_size: int,  # 512 / 640
        # weights: Optional[FasterRCNN_ResNet50_FPN_Weights] = None,
        # trainable_backbone_layers: int = 3,
        # max_det: int = 300,

        num_classes_with_bg=int(cfg.num_classes_with_bg())
        img_size=int(cfg.data.img_size)
        weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        trainable_backbone_layers=3
        max_det=600

        base = fasterrcnn_resnet50_fpn(weights=weights, trainable_backbone_layers=trainable_backbone_layers)
        in_features = base.roi_heads.box_predictor.cls_score.in_features
        base.roi_heads.box_predictor = FastRCNNPredictor(in_features, int(num_classes_with_bg))

        self.backbone = base.backbone
        self.rpn = base.rpn
        self.roi_heads = base.roi_heads
        self.transform = base.transform

        self.transform.min_size = (img_size,)
        self.transform.max_size = img_size
        self.max_det = max_det

    def forward(
        self,
        images: Union[torch.Tensor, Sequence[torch.Tensor]],
        targets: Optional[List[Dict[str, torch.Tensor]]] = None,
    ) -> Tuple[List[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]]:
        x_list = self.as_image_list(images)
        original_sizes = [im.shape[-2:] for im in x_list]

        if targets is not None:
            images_t, targets_t = self.transform(x_list, targets)
        else:
            images_t, targets_t = self.transform(x_list, None)

        feats = self.backbone(images_t.tensors)
        if not isinstance(feats, dict):
            raise TypeError("Backbone must return dict[str, Tensor].")

        proposals, rpn_losses = self.rpn(images_t, feats, targets_t)
        detections, roi_losses = self.roi_heads(feats, proposals, images_t.image_sizes, targets_t)
        outputs = self.transform.postprocess(detections, images_t.image_sizes, original_sizes)

        loss_dict: Dict[str, torch.Tensor] = {}
        if targets is not None:
            loss_dict.update(rpn_losses)
            loss_dict.update(roi_losses)
            loss_dict["total"] = sum(loss_dict.values())

        return outputs, loss_dict
    
    def as_image_list(
        self, 
        images: Union[torch.Tensor, Sequence[torch.Tensor]]
    ) -> List[torch.Tensor]:
        if torch.is_tensor(images):
            x = images
            if x.ndim != 4:
                raise ValueError("Expected NCHW tensor.")
            return [x[i] for i in range(x.shape[0])]
        if len(images) == 0:
            raise ValueError("Empty image list.")
        return list(images)

    # @staticmethod
    # def pack_detections(
    #     detections: List[Dict[str, torch.Tensor]],
    #     max_det: int, device: torch.device,
    # ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    #     N = len(detections)
    #     M = int(max_det)

    #     boxes_b = torch.zeros((N, M, 4), device=device, dtype=torch.float32)
    #     labels_b = torch.full((N, M), -1, device=device, dtype=torch.long)
    #     scores_b = torch.zeros((N, M), device=device, dtype=torch.float32)
    #     valid_b = torch.zeros((N, M), device=device, dtype=torch.bool)

    #     for i, det in enumerate(detections):
    #         if not det:
    #             continue

    #         boxes = det.get("boxes", None)
    #         labels = det.get("labels", None)
    #         scores = det.get("scores", None)

    #         if boxes is None or labels is None or scores is None:
    #             continue
    #         if boxes.numel() == 0:
    #             continue

    #         k = min(M, boxes.shape[0])
    #         boxes_b[i, :k] = boxes[:k].to(device=device, dtype=torch.float32)
    #         labels_b[i, :k] = labels[:k].to(device=device, dtype=torch.long)
    #         scores_b[i, :k] = scores[:k].to(device=device, dtype=torch.float32)
    #         valid_b[i, :k] = True

    #     return boxes_b, labels_b, scores_b, valid_b

    # def loss(
    #     self,
    #     images: Union[torch.Tensor, Sequence[torch.Tensor]],
    #     targets: List[Dict[str, torch.Tensor]],
    # ) -> Dict[str, torch.Tensor]:
    #     x_list = self.as_image_list(images)
    #     # x_list = [im.to(self.device, non_blocking=True) for im in x_list]
    #     # targets = [{k: v.to(self.device, non_blocking=True) for k, v in t.items()} for t in targets]

    #     images_t, targets_t = self.transform(x_list, targets)
    #     feats = self.backbone(images_t.tensors)
    #     if not isinstance(feats, dict):
    #         raise TypeError("Backbone must return dict[str, Tensor].")

    #     proposals, rpn_losses = self.rpn(images_t, feats, targets_t)
    #     _, roi_losses = self.roi_heads(feats, proposals, images_t.image_sizes, targets_t)

    #     out: Dict[str, torch.Tensor] = {}
    #     out.update(rpn_losses)
    #     out.update(roi_losses)
    #     print()
    #     print(rpn_losses)
    #     print(roi_losses)
    #     print()
    #     return out


class FasterRCNNResNet50FPNReloaded(nn.Module):
    def __init__(self, cfg: ExperimentConfig) -> None:
        super().__init__()
        layers_requires_grad = False
        num_classes_with_bg = int(cfg.num_classes_with_bg())
        img_size = int(cfg.data.img_size)

        resnet_model = resnet50(ResNet50_Weights.IMAGENET1K_V2)
        modules = list(resnet_model.children())[:-2]
        self.backbone = nn.Sequential(*modules)
        for param in self.backbone.parameters():
            param.requires_grad = layers_requires_grad

        # RPN
        self.anchor_generator = AnchorGenerator(
            sizes=((img_size // 16, img_size // 8, img_size // 4, img_size // 2),),
            aspect_ratios=((0.5, 1.0, 2.0),)
        )
        self.rpn = RegionProposalNetwork(
            anchor_generator=self.anchor_generator,
            head=RPNHead(2048, self.anchor_generator.num_anchors_per_location()[0]),
            fg_iou_thresh=0.7,
            bg_iou_thresh=0.3,
            batch_size_per_image=256,
            positive_fraction=0.5,
            pre_nms_top_n={"training": 1000, "testing": 500},
            post_nms_top_n={"training": 1000, "testing": 500},
            nms_thresh=0.7,
            score_thresh=0.0
        )

        # ROI pooling
        self.roi_pool = MultiScaleRoIAlign(
            featmap_names=['0'],
            output_size=7,
            sampling_ratio=2
        )

        self.mlp = nn.Sequential(
            nn.Linear(2048 * 7 * 7, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU()
        )

        

        # Predictor
        self.box_predictor = FastRCNNPredictor(
            in_channels=1024,
            num_classes=num_classes_with_bg
        )

    def forward(
        self,
        images: Union[torch.Tensor, Sequence[torch.Tensor]],
        targets: Optional[List[Dict[str, torch.Tensor]]] = None,
    ) -> Tuple[List[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]]:
        x_list = self.as_image_list(images)  

        # 2. Compute backbone features
        # Stack for backbone
        features_tensor = torch.stack(x_list)
        feats = self.backbone(features_tensor)  # [N, C, H, W]
        features = {"0": feats}  # dict for RPN / ROI

        # 3. Wrap images in ImageList for RPN
        original_sizes = [im.shape[-2:] for im in x_list]
        images_list = ImageList(features_tensor, original_sizes)

        # 4. Generate proposals with RPN
        proposals, rpn_losses = self.rpn(images_list, features, targets)

        # 5. ROI pooling
        pooled_features = self.roi_pool(features, proposals, image_shapes=original_sizes)
        box_features = self.mlp(torch.flatten(pooled_features, start_dim=1))
        class_logits, box_regression = self.box_predictor(box_features)  
        pair_class_logits_box_regression = [(self.box_predictor(f)[0], self.box_predictor(f)[1]) for f in box_features] 
        # 7. Construct output detections
        outputs = []
        for props, cls_logits in zip(box_regression, class_logits):
            if props.numel() == 0:
                outputs.append({"boxes": props, "labels": torch.zeros((0,), dtype=torch.int64, device=props.device),
                                "scores": torch.zeros((0,), device=props.device)})
                continue

            scores = cls_logits.softmax(dim=1).max(dim=1).values
            labels = cls_logits.argmax(dim=1)
            outputs.append({"boxes": props, "labels": labels, "scores": scores})

        # 8. Compute loss only if targets are provided
        loss_dict = {}
        if targets is not None:
            labels_list = [t["labels"] for t in targets]
            regression_targets_list = [t["boxes"] for t in targets]

            loss_dict = fastrcnn_loss(class_logits[0], box_regression[0], labels_list, regression_targets_list)
            loss_dict.update(rpn_losses)
            loss_dict["total"] = sum(loss_dict.values())

        return outputs, loss_dict
    
    def as_image_list(
        self, 
        images: Union[torch.Tensor, Sequence[torch.Tensor]]
    ) -> List[torch.Tensor]:
        if torch.is_tensor(images):
            x = images
            if x.ndim != 4:
                raise ValueError("Expected NCHW tensor.")
            return [x[i] for i in range(x.shape[0])]
        if len(images) == 0:
            raise ValueError("Empty image list.")
        return list(images)
