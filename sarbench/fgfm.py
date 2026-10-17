"""Foreground-guided feature modulation (FGFM, from FGAA-FPN: Ma et al., 2026, arXiv 2602.10710; code
github.com/sugmudy/FGAA-FPN, Apache-2.0) in torchvision's FPN. On chosen levels, a small head predicts where objects
are (supervised by "inside a box" masks) and a weight net turns that map into per-channel, per-pixel weights w in
[0, 1]; the lateral map becomes x * (1 + alpha * w) before the top-down sum: foreground boosted, background kept.

Changes from the released code: logits and BCE-with-logits (bf16 autocast forbids BCE on probabilities); a box marks
the cells whose centres it covers plus the cell of its own centre, so a box smaller than one cell still has a
positive (the original rounds the corners and drops boxes that collapse to zero area); the loss is added in
SwinMultiTaskModel (the repository does not include its detector hook)."""
from collections import OrderedDict

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.ops import FeaturePyramidNetwork


class ForegroundHead(nn.Module):
    """FGSegHead: 3 x 3 conv to 64 channels, ReLU, 3 x 3 conv to 1: foreground logits [B, 1, H, W]."""

    def __init__(self, channels=256, mid_channels=64):
        super().__init__()
        self.layers = nn.Sequential(nn.Conv2d(channels, mid_channels, 3, padding=1), nn.ReLU(),
                                    nn.Conv2d(mid_channels, 1, 3, padding=1))

    def forward(self, x):
        return self.layers(x)


class ForegroundWeightNet(nn.Module):
    """FGWeightNet: the foreground map m, optionally sharpened by three learned scalars (all 0 at the start, so m
    passes unchanged), concatenated with the features -> 3 x 3 conv, GroupNorm, ReLU, 1 x 1 conv, sigmoid: weights
    [B, C, H, W] in [0, 1]."""

    def __init__(self, channels=256):
        super().__init__()
        self.layers = nn.Sequential(nn.Conv2d(channels + 1, channels, 3, padding=1), nn.GroupNorm(32, channels),
                                    nn.ReLU(), nn.Conv2d(channels, channels, 1))
        self.sharp_k = nn.Parameter(torch.tensor(0.0))  # slope of the sharpening sigmoid
        self.sharp_b = nn.Parameter(torch.tensor(0.0))  # shift of its threshold from 0.5
        self.mix = nn.Parameter(torch.tensor(0.0))  # 0: m as predicted, 1: fully sharpened

    def forward(self, x, m):
        k, b, mix = self.sharp_k.clamp(-10, 10), self.sharp_b.clamp(-0.4, 0.4), self.mix.clamp(0, 1)
        m = m + mix * (torch.sigmoid(k * (m - 0.5 - b)) - m)
        return torch.sigmoid(self.layers(torch.cat([x, m.to(x.dtype)], dim=1)))


class ForegroundFPN(FeaturePyramidNetwork):
    """torchvision's FPN with FGFM on the lateral maps of `levels` (0: stride 4 ... 3: stride 32 for Swin).
    forward returns (the FPN maps as FeaturePyramidNetwork does, {level name: foreground logits}).
    The head sees detached features and the weight net a detached map, as in the original: the foreground loss
    trains only the head, the detection losses train the weight net (and everything before the lateral convs)."""

    def __init__(self, in_channels_list, out_channels, levels, alpha=0.8, extra_blocks=None):
        super().__init__(in_channels_list, out_channels, extra_blocks=extra_blocks)
        self.levels = sorted(levels)
        self.alpha = alpha
        self.heads = nn.ModuleDict({str(i): ForegroundHead(out_channels) for i in self.levels})
        self.weight_nets = nn.ModuleDict({str(i): ForegroundWeightNet(out_channels) for i in self.levels})

    def forward(self, x):
        names, x = list(x.keys()), list(x.values())
        mask_logits = OrderedDict()

        def lateral(idx):  # the lateral 1 x 1 conv of level idx, modulated if FGFM is on there
            inner = self.get_result_from_inner_blocks(x[idx], idx)
            if idx in self.levels:
                logits = self.heads[str(idx)](inner.detach())
                mask_logits[names[idx]] = logits
                inner = inner * (1 + self.alpha * self.weight_nets[str(idx)](inner, logits.detach().sigmoid()))
            return inner

        # From here on as FeaturePyramidNetwork.forward, with lateral() for the lateral maps
        last = len(x) - 1
        last_inner = lateral(last)
        results = [self.get_result_from_layer_blocks(last_inner, last)]
        for idx in range(last - 1, -1, -1):
            inner_lateral = lateral(idx)
            last_inner = inner_lateral + F.interpolate(last_inner, size=inner_lateral.shape[-2:], mode='nearest')
            results.insert(0, self.get_result_from_layer_blocks(last_inner, idx))
        if self.extra_blocks is not None:
            results, names = self.extra_blocks(results, x, names)
        return OrderedDict(zip(names, results)), mask_logits


def box_masks(targets, size, stride):
    """Per image a [size, size] float map: 1 on the cells whose centres lie inside a box (XYXY, input pixels), and
    on the cell of every box centre, so that boxes smaller than a cell are not lost; 0 elsewhere."""
    centres = (torch.arange(size, device=targets[0]['boxes'].device) + 0.5) * stride  # of the cells, in input pixels
    masks = []
    for t in targets:
        x1, y1, x2, y2 = t['boxes'].float().unbind(dim=1)
        in_x = (centres[None] >= x1[:, None]) & (centres[None] <= x2[:, None])  # [boxes, size]
        in_y = (centres[None] >= y1[:, None]) & (centres[None] <= y2[:, None])
        mask = (in_y[:, :, None] & in_x[:, None, :]).any(dim=0)  # [size, size]; all False without boxes
        cx = ((x1 + x2) / 2 / stride).long().clamp(0, size - 1)
        cy = ((y1 + y2) / 2 / stride).long().clamp(0, size - 1)
        mask[cy, cx] = True
        masks.append(mask.float())
    return torch.stack(masks)


def foreground_loss(logits, target, dice_weight=0.6, max_pos_weight=20.0):
    """The original's loss, for one level: BCE with the foreground cells weighted by background / foreground
    (at most 20), plus 0.6 x Dice per image. logits [B, 1, H, W], target [B, H, W]."""
    logits = logits.squeeze(1).float()
    positives = target.sum()
    pos_weight = ((target.numel() - positives) / positives.clamp(min=1)).clamp(max=max_pos_weight)
    bce = F.binary_cross_entropy_with_logits(logits, target, weight=1 + (pos_weight - 1) * target)
    p = logits.sigmoid()
    dice = 1 - (2 * (p * target).sum(dim=(1, 2)) + 1e-6) / (p.sum(dim=(1, 2)) + target.sum(dim=(1, 2)) + 1e-6)
    return bce + dice_weight * dice.mean()
