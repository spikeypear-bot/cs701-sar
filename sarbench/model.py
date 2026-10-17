"""The multi-task model: one shared ViT backbone feeds a classification head and a Faster R-CNN detector."""
from collections import OrderedDict

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, TwoMLPHead
from torchvision.models.detection.image_list import ImageList
from torchvision.models.detection.roi_heads import RoIHeads
from torchvision.models.detection.rpn import RegionProposalNetwork, RPNHead
from torchvision.ops import FeaturePyramidNetwork, MultiScaleRoIAlign
from torchvision.ops.feature_pyramid_network import LastLevelMaxPool

from .adapters import set_task


class SimpleFeaturePyramid(nn.Module):
    """ViTDet neck (Li et al., 2022): the single stride-16 ViT map -> 256-channel maps at strides 4, 8, 16, 32, 64."""

    def __init__(self, dim=768, channels=256):
        super().__init__()
        self.rescale = nn.ModuleList([
            nn.Sequential(nn.ConvTranspose2d(dim, dim // 2, 2, stride=2), nn.GroupNorm(32, dim // 2), nn.GELU(),
                          nn.ConvTranspose2d(dim // 2, dim // 4, 2, stride=2)),  # stride 4
            nn.ConvTranspose2d(dim, dim // 2, 2, stride=2),  # stride 8
            nn.Identity(),  # stride 16
            nn.MaxPool2d(2),  # stride 32
        ])
        self.project = nn.ModuleList(
            nn.Sequential(nn.Conv2d(c, channels, 1, bias=False), nn.GroupNorm(32, channels),
                          nn.Conv2d(channels, channels, 3, padding=1, bias=False), nn.GroupNorm(32, channels))
            for c in (dim // 4, dim // 2, dim, dim))

    def forward(self, x):
        levels = [project(rescale(x)) for rescale, project in zip(self.rescale, self.project)]
        levels.append(F.max_pool2d(levels[-1], 1, stride=2))  # stride 64, as torchvision's FPN LastLevelMaxPool
        return OrderedDict(zip(['0', '1', '2', '3', 'pool'], levels))


def gaussian_heatmaps(targets, size, stride, min_sigma):
    """Object-centre targets (after TRANSAR, Almalioglu et al., 2025): per image a [size, size] map with a Gaussian at
    every box centre, 1 at the centre, sigma = sqrt(w h) / 6 input pixels (at least min_sigma), so it scales with the
    object. Overlapping Gaussians take the max: close objects keep separate peaks instead of adding up past 1."""
    centres = (torch.arange(size, device=targets[0]['boxes'].device) + 0.5) * stride  # of the cells, in input pixels
    maps = []
    for t in targets:
        x1, y1, x2, y2 = t['boxes'].float().unbind(dim=1)
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        sigma = ((x2 - x1) * (y2 - y1)).clamp(min=0).sqrt().div(6).clamp(min=min_sigma)
        dx2 = (centres[None] - cx[:, None]) ** 2  # [boxes, size]
        dy2 = (centres[None] - cy[:, None]) ** 2
        blobs = torch.exp(-(dy2[:, :, None] + dx2[:, None, :]) / (2 * sigma[:, None, None] ** 2))  # [boxes, y, x]
        maps.append(blobs.amax(dim=0) if len(blobs) else centres.new_zeros(size, size))
    return torch.stack(maps)


def heatmap_loss(logits, target, bce_weight=0.05, dice_weight=1.0):
    """TRANSAR's BCE + Dice. Over 99 % of the cells are background: BCE alone is lowest for an empty map, while Dice
    (the overlap of prediction and target, per image) does not depend on the amount of background."""
    bce = F.binary_cross_entropy_with_logits(logits, target)
    p = logits.sigmoid()
    dice = 1 - (2 * (p * target).sum(dim=(1, 2)) + 1) / (p.sum(dim=(1, 2)) + target.sum(dim=(1, 2)) + 1)
    return bce_weight * bce + dice_weight * dice.mean()


class MultiTaskModel(nn.Module):
    """backbone tokens -> classification logits (mean token -> LayerNorm -> Linear) and Faster R-CNN detections.
    heatmap=True adds an auxiliary head for training only: object-centre heatmaps on the stride-4 level '0', the level
    of the smallest anchors, so every object, however small, sends a dense signal into the shared features."""

    def __init__(self, backbone, task_routing: bool, num_classes=9, heatmap=False, heatmap_min_sigma=2.0):
        super().__init__()
        self.backbone = backbone
        self.task_routing = task_routing  # MoE-LoRA: the backbone runs once per task, with that task's routers
        self.cls_head = nn.Sequential(nn.LayerNorm(backbone.embed_dim), nn.Linear(backbone.embed_dim, num_classes))
        self.neck = SimpleFeaturePyramid(backbone.embed_dim)
        # Small objects (median ~16 px at 512 px, 99 % below ~240 px): two anchor scales per octave, from 8 px at
        # stride 4 up to 181 px at stride 64, times 3 aspect ratios = 6 anchors per location.
        anchors = AnchorGenerator(((8, 11), (16, 23), (32, 45), (64, 91), (128, 181)), ((0.5, 1.0, 2.0),) * 5)
        # Everything else is torchvision's Faster R-CNN default.
        self.rpn = RegionProposalNetwork(
            anchors, RPNHead(256, 6), fg_iou_thresh=0.7, bg_iou_thresh=0.3, batch_size_per_image=256,
            positive_fraction=0.5, pre_nms_top_n=dict(training=2000, testing=1000),
            post_nms_top_n=dict(training=2000, testing=1000), nms_thresh=0.7)
        self.roi_heads = RoIHeads(
            MultiScaleRoIAlign(['0', '1', '2', '3'], output_size=7, sampling_ratio=2),
            TwoMLPHead(256 * 7 * 7, 1024), FastRCNNPredictor(1024, num_classes + 1),  # + 1: background
            fg_iou_thresh=0.5, bg_iou_thresh=0.5, batch_size_per_image=512, positive_fraction=0.25,
            bbox_reg_weights=None, score_thresh=0.05, nms_thresh=0.5, detections_per_img=100)
        # Unused at inference: predict.py builds the model without it and skips its weights
        self.heatmap_head = nn.Sequential(nn.Conv2d(256, 128, 3, padding=1), nn.ReLU(),
                                          nn.Conv2d(128, 1, 1)) if heatmap else None
        self.heatmap_min_sigma = heatmap_min_sigma

    def heatmap_losses(self, images, features, targets):
        """{'heatmap': loss} when training with the heatmap head, else {}."""
        if not (self.training and self.heatmap_head is not None):
            return {}
        logits = self.heatmap_head(features['0']).squeeze(1).float()  # [B, 128, 128] at 512 px
        stride = images.shape[-1] // logits.shape[-1]  # square inputs (data.py resizes to size x size)
        target = gaussian_heatmaps(targets, logits.shape[-1], stride, self.heatmap_min_sigma)
        return {'heatmap': heatmap_loss(logits, target)}

    def forward(self, images, targets=None):
        """images [B, 1, H, W], targets: per-image {'boxes' XYXY, 'labels' 1..9} (training only) ->
        (logits [B, num_classes], per-image detections (eval mode), the four Faster R-CNN losses (training mode),
        plus 'heatmap' when training with the heatmap head)."""
        set_task(self.backbone, 0)  # 0 = classification, 1 = detection; a no-op without MoE-LoRA
        tokens = self.backbone(images)
        logits = self.cls_head(tokens.mean(dim=1))
        if self.task_routing:
            set_task(self.backbone, 1)
            tokens = self.backbone(images)
        h, w = images.shape[-2] // 16, images.shape[-1] // 16  # one token per 16 x 16 patch
        features = self.neck(tokens.transpose(1, 2).unflatten(2, (h, w)))  # tokens as a [B, 768, h, w] map
        heatmap = self.heatmap_losses(images, features, targets)  # bf16 features, before the fp32 detector
        # Precision rule: the box coder casts anchors to the dtype of the regression output, so in bf16 every
        # coordinate near 512 would snap to a 2 px grid. RPN and RoIHeads therefore run in fp32.
        with torch.autocast(images.device.type, enabled=False):
            features = {name: f.float() for name, f in features.items()}
            image_list = ImageList(images, [tuple(images.shape[-2:])] * len(images))
            proposals, rpn_losses = self.rpn(image_list, features, targets)
            detections, roi_losses = self.roi_heads(features, proposals, image_list.image_sizes, targets)
        return logits, detections, {**rpn_losses, **roi_losses, **heatmap}


class SwinMultiTaskModel(MultiTaskModel):
    """MultiTaskModel on a Swin backbone: its 4 real maps (strides 4-32) go through an FPN instead of the ViTDet neck.
    The classification head, anchors, RPN and RoI heads are MultiTaskModel's, unchanged."""

    def __init__(self, backbone, task_routing: bool, num_classes=9, **heatmap_options):
        # cls_head on backbone.embed_dim, anchors, rpn, roi_heads, heatmap_head
        super().__init__(backbone, task_routing, num_classes, **heatmap_options)
        # Replaces the ViTDet neck. LastLevelMaxPool adds the stride-64 'pool' level that the anchors expect.
        self.neck = FeaturePyramidNetwork(backbone.channels, 256, extra_blocks=LastLevelMaxPool())

    def forward(self, images, targets=None):
        """As MultiTaskModel.forward, with 4 maps from the backbone instead of tokens."""
        set_task(self.backbone, 0)
        maps = self.backbone(images)  # 4 maps [B, C_i, H_i, W_i]
        logits = self.cls_head(maps[-1].mean(dim=(2, 3)))  # global average of the last map -> [B, embed_dim]
        if self.task_routing:
            set_task(self.backbone, 1)
            maps = self.backbone(images)
        features = self.neck(OrderedDict((str(i), m) for i, m in enumerate(maps)))  # '0'..'3' + 'pool', 256 channels
        heatmap = self.heatmap_losses(images, features, targets)
        # From here on as in MultiTaskModel.forward: the detector in fp32 (see the precision rule there)
        with torch.autocast(images.device.type, enabled=False):
            features = {name: f.float() for name, f in features.items()}
            image_list = ImageList(images, [tuple(images.shape[-2:])] * len(images))
            proposals, rpn_losses = self.rpn(image_list, features, targets)
            detections, roi_losses = self.roi_heads(features, proposals, image_list.image_sizes, targets)
        return logits, detections, {**rpn_losses, **roi_losses, **heatmap}
