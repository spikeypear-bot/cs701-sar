"""sarbench/fgfm.py on random maps (CPU, no data): box masks, the foreground loss, ForegroundFPN's outputs, and which
part each loss trains."""
from collections import OrderedDict

import torch
from torchvision.ops import FeaturePyramidNetwork
from torchvision.ops.feature_pyramid_network import LastLevelMaxPool

from sarbench.fgfm import ForegroundFPN, box_masks, foreground_loss

CHANNELS = [16, 32, 64, 128]  # a small Swin: maps at strides 4, 8, 16, 32 of a 128 px input


def swin_like_maps(batch=2):
    return OrderedDict((str(i), torch.randn(batch, c, 32 >> i, 32 >> i)) for i, c in enumerate(CHANNELS))


def small_fpn(levels=(0, 1, 2)):
    return ForegroundFPN(CHANNELS, 32, levels, extra_blocks=LastLevelMaxPool())


def test_box_masks_cover_the_cells_inside_each_box():
    # A 16 px box over the stride-4 cells 10-13 (centres 42-54 px) on rows and columns
    mask, empty = box_masks([{'boxes': torch.tensor([[40., 40., 56., 56.]])}, {'boxes': torch.zeros(0, 4)}], 128, 4)
    assert mask.shape == (128, 128) and mask.sum() == 16
    assert mask[10:14, 10:14].all()
    assert (empty == 0).all()


def test_box_masks_keep_boxes_smaller_than_a_cell():
    # A 4 px box covers no stride-16 cell centre; the cell of its centre (row 2, col 3) is still foreground
    mask = box_masks([{'boxes': torch.tensor([[50., 34., 54., 38.]])}], 32, 16)[0]
    assert mask.sum() == 1 and mask[2, 3] == 1


def test_foreground_loss_is_lower_for_the_target_than_for_an_empty_map():
    target = box_masks([{'boxes': torch.tensor([[40., 40., 72., 72.]])}], 128, 4)
    right = torch.logit(target.clamp(1e-4, 1 - 1e-4))[:, None]
    assert foreground_loss(right, target) < foreground_loss(right * 0 - 10, target)


def test_outputs_match_the_plain_fpn_and_masks_come_from_the_chosen_levels():
    maps = swin_like_maps()
    plain = FeaturePyramidNetwork(CHANNELS, 32, extra_blocks=LastLevelMaxPool())(maps)
    features, mask_logits = small_fpn()(maps)
    assert list(features) == list(plain) == ['0', '1', '2', '3', 'pool']
    assert all(features[k].shape == plain[k].shape for k in plain)
    assert list(mask_logits) == ['2', '1', '0']  # computed top-down, as the lateral maps
    assert all(mask_logits[k].shape == (2, 1, *features[k].shape[-2:]) for k in mask_logits)


def test_mask_loss_trains_only_the_heads_and_detection_trains_the_weight_nets():
    fpn = small_fpn()
    features, mask_logits = fpn(swin_like_maps())
    targets = [{'boxes': torch.tensor([[8., 8., 40., 40.]])}] * 2
    mask_loss = sum(foreground_loss(logits, box_masks(targets, logits.shape[-1], 128 // logits.shape[-1]))
                    for logits in mask_logits.values())
    mask_grads = torch.autograd.grad(mask_loss, list(fpn.parameters()), allow_unused=True)
    names = [n for n, _ in fpn.named_parameters()]
    assert {n.split('.')[0] for n, g in zip(names, mask_grads) if g is not None} == {'heads'}

    detection_loss = sum(f.square().mean() for f in features.values())
    detection_grads = torch.autograd.grad(detection_loss, list(fpn.parameters()), allow_unused=True)
    trained = {n.split('.')[0] for n, g in zip(names, detection_grads) if g is not None and g.abs().sum() > 0}
    assert 'heads' not in trained and {'weight_nets', 'inner_blocks', 'layer_blocks'} <= trained
