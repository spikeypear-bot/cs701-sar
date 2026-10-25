"""The multi-task model: one shared ViT backbone feeds a classification head and a detector (Faster R-CNN or
RetinaNet)."""
from collections import OrderedDict
from functools import partial

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models.detection import _utils as det_utils
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, TwoMLPHead
from torchvision.models.detection.image_list import ImageList
from torchvision.models.detection.retinanet import RetinaNet, RetinaNetHead
from torchvision.models.detection.roi_heads import RoIHeads
from torchvision.models.detection.rpn import RegionProposalNetwork, RPNHead
from torchvision.ops import FeaturePyramidNetwork, MultiScaleRoIAlign, box_iou
from torchvision.ops.feature_pyramid_network import LastLevelMaxPool

from .adapters import set_task
from .fgfm import ForegroundFPN, box_masks, foreground_loss

RPN_MATCH = ('iou', 'nwd', 'max')  # the choices of rpn_match, see anchor_similarity
DETECTORS = ('faster_rcnn', 'retinanet')  # the choices of detector


def anchor_similarity(gt_boxes, anchors, mode, c):
    """[N, 4], [M, 4] XYXY -> [N, M], the match quality the RPN labels anchors by (positive >= 0.7, background < 0.3).
    'iou': torchvision's default; 'nwd': normalized Gaussian Wasserstein distance (Wang et al., 2021, arXiv
    2110.13389): each box as a 2D Gaussian (mean its centre, std half its size), exp(-W2 / c), W2 the Euclidean
    distance of the (cx, cy, w / 2, h / 2) vectors. A shift of d px costs the same for every box size, while IoU falls
    fastest for the smallest boxes: with 8 px anchors every 4 px, an 8 px object can be 2 px off its nearest anchor
    in x and y: IoU 0.39, NWD 0.84 at c = 16. 'max': the larger of the two, NWD for tiny boxes and IoU for large
    ones, which NWD's fixed c makes strict (a 200 px box 16 px off in x and y: IoU 0.73, NWD 0.24 at c = 16)."""
    if mode == 'iou':
        return box_iou(gt_boxes, anchors)

    def gaussian(b):
        return torch.stack([(b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2,
                            (b[:, 2] - b[:, 0]) / 2, (b[:, 3] - b[:, 1]) / 2], dim=1)

    # Exact differences: cdist's matrix-product shortcut errs by up to ~0.2 px for close boxes near 512 px in fp32
    w2 = torch.cdist(gaussian(gt_boxes), gaussian(anchors), compute_mode='donot_use_mm_for_euclid_dist')
    nwd = torch.exp(-w2 / c)
    return nwd if mode == 'nwd' else torch.maximum(nwd, box_iou(gt_boxes, anchors))


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


class RetinaNetDetector(nn.Module):
    """RetinaNet (Lin et al., 2017) in place of the RPN + RoI heads: torchvision's head (4 convs per branch, focal
    loss, L1 box loss), anchor matching (IoU >= 0.5 object, < 0.4 background, every box keeps its best anchor) and
    post-processing, on the same anchors and FPN maps. torchvision's RetinaNet module also resizes and normalizes the
    images, which data.py does already, so only its detector methods are borrowed here."""
    compute_loss = RetinaNet.compute_loss  # reads self.head and self.proposal_matcher
    postprocess_detections = RetinaNet.postprocess_detections  # self.box_coder and the four thresholds below

    def __init__(self, anchors, num_classes):
        super().__init__()
        self.anchor_generator = anchors
        self.num_anchors = anchors.num_anchors_per_location()[0]  # per location, the same on every level
        # One column per class, without background (sigmoid focal loss): labels 1..9 are columns 0..8
        self.head = RetinaNetHead(256, self.num_anchors, num_classes)
        self.proposal_matcher = det_utils.Matcher(0.5, 0.4, allow_low_quality_matches=True)
        self.box_coder = det_utils.BoxCoder(weights=(1.0, 1.0, 1.0, 1.0))
        # torchvision's defaults, except 100 detections per image (300 there), as the Faster R-CNN and COCO mAP
        self.score_thresh, self.nms_thresh, self.detections_per_img, self.topk_candidates = 0.05, 0.5, 100, 1000

    def forward(self, image_list, features, targets=None):
        """fp32 features {name: [B, 256, h, w]} -> (per-image detections with labels 1..9 (eval mode),
        {'loss_focal', 'loss_retina_box_reg'} (training mode)).
        Unlike the RPN head (one conv), this head is 8 convs on every level, the stride-4 one included: in fp32 it
        would cost more than the backbone. So it runs in the caller's bf16, and only its outputs go to fp32, before
        they meet the anchors (see the precision rule in MultiTaskModel.detect)."""
        features = list(features.values())
        head_outputs = self.head(features)
        with torch.autocast(image_list.tensors.device.type, enabled=False):
            head_outputs = {name: out.float() for name, out in head_outputs.items()}
            anchors = self.anchor_generator(image_list, features)  # fp32 like the features
            if self.training:
                targets = [{**t, 'labels': t['labels'] - 1} for t in targets]  # column of the class
                losses = self.compute_loss(targets, head_outputs, anchors)
                # Not 'classification': train.py uses that name for the image class
                return [], {'loss_focal': losses['classification'], 'loss_retina_box_reg': losses['bbox_regression']}
            per_level = [f.shape[-2] * f.shape[-1] * self.num_anchors for f in features]
            head_outputs = {name: list(out.split(per_level, dim=1)) for name, out in head_outputs.items()}
            anchors = [list(a.split(per_level)) for a in anchors]
            detections = self.postprocess_detections(head_outputs, anchors, image_list.image_sizes)
        return [{**d, 'labels': d['labels'] + 1} for d in detections], {}


class MultiTaskModel(nn.Module):
    """backbone tokens -> classification logits (mean token -> LayerNorm -> Linear) and detections.
    detector: 'faster_rcnn' (RPN + RoI heads) or 'retinanet' (RetinaNetDetector), on the same anchors.
    heatmap=True adds an auxiliary head for training only: object-centre heatmaps on the stride-4 level '0', the level
    of the smallest anchors, so every object, however small, sends a dense signal into the shared features.
    rpn_match ('iou', 'nwd' or 'max', see anchor_similarity) is how the RPN labels anchors in training, nwd_c the
    constant of NWD in input pixels; neither matters at inference, nor with RetinaNet."""

    def __init__(self, backbone, task_routing: bool, num_classes=9, heatmap=False, heatmap_min_sigma=2.0,
                 rpn_match='iou', nwd_c=16.0, detector='faster_rcnn', image_size=512):
        super().__init__()
        self.backbone = backbone
        self.task_routing = task_routing  # MoE-LoRA: the backbone runs once per task, with that task's routers
        self.cls_head = nn.Sequential(nn.LayerNorm(backbone.embed_dim), nn.Linear(backbone.embed_dim, num_classes))
        self.neck = SimpleFeaturePyramid(backbone.embed_dim)
        # Small objects (median ~16 px at 512 px, 99 % below ~240 px): two anchor scales per octave, from 8 px at
        # stride 4 up to 181 px at stride 64, times 3 aspect ratios = 6 anchors per location. At another image_size
        # the anchors scale with the objects, so each object meets the same anchors and only gets more feature cells.
        scale = image_size / 512
        anchors = AnchorGenerator(tuple((a * scale, b * scale) for a, b in ((8, 11), (16, 23), (32, 45), (64, 91),
                                                                           (128, 181))), ((0.5, 1.0, 2.0),) * 5)
        self.detector = detector
        if detector == 'retinanet':
            self.retinanet = RetinaNetDetector(anchors, num_classes)
        else:  # Everything else is torchvision's Faster R-CNN default
            self.rpn = RegionProposalNetwork(
                anchors, RPNHead(256, 6), fg_iou_thresh=0.7, bg_iou_thresh=0.3, batch_size_per_image=256,
                positive_fraction=0.5, pre_nms_top_n=dict(training=2000, testing=1000),
                post_nms_top_n=dict(training=2000, testing=1000), nms_thresh=0.7)
            if rpn_match != 'iou':  # same thresholds as with IoU, as in Wang et al. (2021)
                self.rpn.box_similarity = partial(anchor_similarity, mode=rpn_match, c=nwd_c)
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

    def detect(self, images, features, targets):
        """The detector on the neck's maps -> (per-image detections (eval mode), its losses (training mode)).
        Precision rule: the box coder casts anchors to the dtype of the regression output, so in bf16 every
        coordinate near 512 would snap to a 2 px grid. The detector therefore gets fp32 maps, and RPN and RoIHeads
        run in fp32; RetinaNet's head runs in bf16 and casts its outputs (see RetinaNetDetector.forward)."""
        features = {name: f.float() for name, f in features.items()}
        image_list = ImageList(images, [tuple(images.shape[-2:])] * len(images))
        if self.detector == 'retinanet':
            return self.retinanet(image_list, features, targets)
        with torch.autocast(images.device.type, enabled=False):
            proposals, rpn_losses = self.rpn(image_list, features, targets)
            detections, roi_losses = self.roi_heads(features, proposals, image_list.image_sizes, targets)
        return detections, {**rpn_losses, **roi_losses}

    def forward(self, images, targets=None):
        """images [B, 1, H, W], targets: per-image {'boxes' XYXY, 'labels' 1..9} (training only) ->
        (logits [B, num_classes], per-image detections (eval mode), the detector's losses (training mode),
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
        detections, det_losses = self.detect(images, features, targets)
        return logits, detections, {**det_losses, **heatmap}


class SwinMultiTaskModel(MultiTaskModel):
    """MultiTaskModel on a Swin or ConvNeXt backbone: its 4 real maps (strides 4-32) go through an FPN instead of the
    ViTDet neck.
    The classification head, anchors and detector are MultiTaskModel's, unchanged. fgfm_levels (e.g. [0, 1, 2])
    puts foreground-guided feature modulation on those FPN levels (sarbench/fgfm.py); it runs at inference too."""

    def __init__(self, backbone, task_routing: bool, num_classes=9, fgfm_levels=(), fgfm_alpha=0.8, **options):
        # cls_head on backbone.embed_dim, the detector, heatmap_head; options: heatmap, rpn_match and detector ones
        super().__init__(backbone, task_routing, num_classes, **options)
        # Replaces the ViTDet neck. LastLevelMaxPool adds the stride-64 'pool' level that the anchors expect.
        if fgfm_levels:
            self.neck = ForegroundFPN(backbone.channels, 256, fgfm_levels, fgfm_alpha, extra_blocks=LastLevelMaxPool())
        else:
            self.neck = FeaturePyramidNetwork(backbone.channels, 256, extra_blocks=LastLevelMaxPool())

    def foreground_losses(self, images, mask_logits, targets):
        """{'foreground': mean over the FGFM levels of foreground_loss} when training with FGFM, else {}. Levels
        without a single box cell in the batch are skipped, as in the original."""
        if not (self.training and mask_logits):
            return {}
        losses = []
        for logits in mask_logits.values():
            stride = images.shape[-1] // logits.shape[-1]  # square inputs (data.py resizes to size x size)
            target = box_masks(targets, logits.shape[-1], stride)
            if target.any():
                losses.append(foreground_loss(logits, target))
        return {'foreground': torch.stack(losses).mean()} if losses else {}

    def forward(self, images, targets=None):
        """As MultiTaskModel.forward, with 4 maps from the backbone instead of tokens, plus 'foreground' when
        training with FGFM."""
        set_task(self.backbone, 0)
        maps = self.backbone(images)  # 4 maps [B, C_i, H_i, W_i]
        logits = self.cls_head(maps[-1].mean(dim=(2, 3)))  # global average of the last map -> [B, embed_dim]
        if self.task_routing:
            set_task(self.backbone, 1)
            maps = self.backbone(images)
        features = self.neck(OrderedDict((str(i), m) for i, m in enumerate(maps)))  # '0'..'3' + 'pool', 256 channels
        features, mask_logits = features if isinstance(self.neck, ForegroundFPN) else (features, {})
        auxiliary = {**self.heatmap_losses(images, features, targets),
                     **self.foreground_losses(images, mask_logits, targets)}
        detections, det_losses = self.detect(images, features, targets)
        return logits, detections, {**det_losses, **auxiliary}
