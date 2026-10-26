"""sarbench/data.py on the real dataset: split contents, the box geometry round trip, and flips; then the
rotations, BoxPaste and the filter channels on synthetic images."""
import random
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import tv_tensors

from sarbench.data import (MEAN, STD, BoxPaste, SARMultiTask, collate, input_channels, jitter_intensity,
                           random_crop, rot90, to_original_xywh)
from sarbench.metrics import detection_metrics

ROOT = Path(__file__).resolve().parents[1] / 'data'  # make_split.py's split, as train.slurm's --data data


def batches(split):
    return DataLoader(SARMultiTask(ROOT, split), batch_size=64, collate_fn=collate)


@pytest.mark.parametrize('split, size', [('train', 8452), ('val', 940), ('test', 940)])
def test_subset_a(split, size):
    loader = batches(split)
    assert len(loader.dataset) == size
    for images, labels, targets in loader:
        assert images.shape[1:] == (1, 512, 512) and images.dtype == torch.float32
        for label, target in zip(labels.tolist(), targets):
            assert len(target['labels']) >= 1
            assert (target['labels'] == label + 1).all()
            assert target['boxes'].min() >= 0 and target['boxes'].max() <= 512


@pytest.mark.parametrize('split', ['val', 'test'])
def test_ground_truth_round_trip_scores_map_1(split):
    predictions, image_ids = [], []
    for _, _, targets in batches(split):
        for t in targets:
            image_ids.append(t['image_id'])
            boxes = to_original_xywh(t['boxes'], t['orig_size']).tolist()
            predictions += [{'image_id': t['image_id'], 'category_id': c, 'bbox': b, 'score': 1.0}
                            for b, c in zip(boxes, t['labels'].tolist())]
    metrics = detection_metrics(ROOT / 'detection' / f'instances_{split}.json', predictions, image_ids)
    assert round(metrics['mAP'], 3) == 1.0
    assert [round(ap, 3) for ap in metrics['per_class_AP'].values()] == [1.0] * 9


def test_flips_move_boxes_with_pixels():
    transform = SARMultiTask(ROOT, 'train', train=True).transform
    image = torch.zeros(1, 200, 300, dtype=torch.uint8)  # non-square, so the resize also stretches it
    image[:, 50:90, 30:120] = 255  # a bright rectangle filling exactly this box:
    box = tv_tensors.BoundingBoxes([[30., 50., 120., 90.]], format='XYXY', canvas_size=(200, 300))
    flips = set()
    for seed in range(8):
        torch.manual_seed(seed)
        out, out_box = transform(image, box)
        ys, xs = torch.nonzero(out[0] * STD + MEAN > 0.5, as_tuple=True)
        bright = torch.stack([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1])  # pixel i spans [i, i + 1)
        assert torch.allclose(bright.float(), out_box[0], atol=1)
        flips.add((out_box[0, 0].item() > 256, out_box[0, 1].item() > 256))  # (horizontal, vertical)
    assert len(flips) == 4  # the seeds cover every combination of the two flips


def bright_box(image):
    """The XYXY extent of the pixels above 0.5 of the first channel; pixel i spans [i, i + 1)."""
    ys, xs = torch.nonzero(image[0] > 0.5, as_tuple=True)
    return torch.stack([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1]).float()


def test_rot90_moves_boxes_with_pixels():
    image = torch.zeros(1, 64, 64)
    image[:, 10:20, 5:40] = 1  # a bright rectangle filling exactly this box:
    box = torch.tensor([[5., 10., 40., 20.]])
    for k in range(4):
        out, out_box = rot90(image, box, k)
        assert torch.equal(bright_box(out), out_box[0])


def test_random_crop_zooms_in_and_moves_the_boxes_with_the_pixels():
    image = torch.zeros(1, 512, 512)
    image[:, 200:240, 100:180] = 1  # box 0, bright
    boxes = torch.tensor([[100., 200., 180., 240.], [0., 0., 8., 8.], [490., 490., 512., 512.]])
    picked_far_box = False
    for seed in range(50):
        torch.manual_seed(seed)
        out, out_boxes, kept = random_crop(image, boxes, min_side=0.5)
        assert out.shape == image.shape and kept.any() and len(out_boxes) == kept.sum()
        assert out_boxes.min() >= 0 and out_boxes.max() <= 512
        if kept[0]:  # the bright box, where its pixels went (zoomed: bilinear edges, so 1 px of slack per side)
            x1, y1, x2, y2 = out_boxes[0].tolist()
            assert out[0, int(y1) + 2:int(y2) - 2, int(x1) + 2:int(x2) - 2].min() > 0.5
            if (out_boxes[0] > 0).all() and (out_boxes[0] < 512).all():  # whole: zoomed in, never out
                assert x2 - x1 >= 80 - 1e-3 and y2 - y1 >= 40 - 1e-3
        picked_far_box |= not kept[0]
    assert picked_far_box  # the crop sometimes holds a corner box and leaves box 0 out


def test_boxpaste_adds_free_boxes_over_the_pasted_objects(tmp_path):
    donor = np.zeros((64, 64), dtype=np.uint8)
    donor[20:28, 20:28] = 200  # an 8 px object at [20, 20, 8, 8]
    Image.fromarray(donor).save(tmp_path / 'donor.png')
    paste = BoxPaste([(tmp_path / 'donor.png', (64, 64), 3, [20., 20., 8., 8.])], prob=1, max_pastes=3, max_size=32)
    gray = np.zeros((64, 64), dtype=np.uint8)
    for seed in range(5):
        random.seed(seed)
        out, boxes = paste(gray, [[0., 0., 10., 10.]], category=3)
        assert boxes[0] == [0., 0., 10., 10.] and 2 <= len(boxes) <= 4
        for i, (x, y, w, h) in enumerate(boxes[1:], 1):
            assert (w, h) == (8., 8.) and (out[int(y):int(y + h), int(x):int(x + w)] == 200).all()
            for x2, y2, w2, h2 in boxes[:i]:  # no overlap with earlier boxes
                assert x + w <= x2 or x2 + w2 <= x or y + h <= y2 or y2 + h2 <= y
    assert paste(gray, [[0., 0., 10., 10.]], category=4)[1] == [[0., 0., 10., 10.]]  # no donors of class 4


def test_jitter_intensity_scales_contrast_and_brightness_within_its_range():
    gray = np.tile(np.arange(0, 200, 2, dtype=np.uint8), (8, 1))  # a ramp 0..198, mean 99
    for seed in range(20):
        torch.manual_seed(seed)
        out = jitter_intensity(gray, 0.2)
        torch.manual_seed(seed)
        c, b = (1 + 0.2 * (2 * torch.rand(2) - 1)).tolist()
        assert 0.8 <= c <= 1.2 and 0.8 <= b <= 1.2 and out.dtype == np.uint8 and out.shape == gray.shape
        expected = (((gray.astype(np.float32) - 99) * c + 99) * b).round().clip(0, 255)
        assert np.array_equal(out, expected.astype(np.uint8))
    assert np.array_equal(jitter_intensity(gray, 0.0), gray)
    assert SARMultiTask(ROOT, 'val', limit=2, brightness=0.2).brightness == 0.0  # not training: no jitter


def test_input_channels_keep_raw_and_mark_bright_points():
    gray = np.full((64, 64), 60, dtype=np.uint8)
    gray[30:33, 30:33] = 250  # a bright 3 px point target
    maps = input_channels(gray, ('raw', 'nlm', 'cv'))
    assert maps.shape == (64, 64, 3) and maps.dtype == np.uint8
    assert (maps[..., 0] == gray).all()
    assert maps[31, 31, 2] > 50 and maps[5, 5, 2] == 0  # coefficient of variation: high at the point, 0 when flat


def test_multilook_lowers_speckle_and_keeps_intensity():
    """3 x 3 multi-look on 1-look speckle (Rayleigh amplitude): std about 3x lower, mean intensity unchanged."""
    rng = np.random.default_rng(0)
    speckled = np.clip(100 * np.sqrt(rng.exponential(1, (64, 64))), 0, 255).astype(np.uint8)
    ml = input_channels(speckled, ('ml',))[..., 0].astype(float)
    assert ml.shape == speckled.shape
    assert ml.std() < 0.5 * speckled.std()
    assert abs((ml ** 2).mean() / (speckled.astype(float) ** 2).mean() - 1) < 0.05


def test_gre_marks_edges_but_not_speckle():
    """Gradient by ratio: high at a step edge, about equal on dark and bright speckled ground (multiplicative noise)."""
    rng = np.random.default_rng(0)
    clean = np.full((64, 64), 40.0)
    clean[:, 32:] = 160  # a step edge at column 32, 4 x brighter
    speckled = np.clip(clean * rng.gamma(16, 1 / 16, clean.shape), 0, 255).astype(np.uint8)
    gre = input_channels(speckled, ('gre',))[..., 0].astype(float)
    edge, dark, bright = gre[10:54, 31:33].mean(), gre[10:54, 8:20].mean(), gre[10:54, 44:56].mean()
    assert edge > 3 * max(dark, bright)
    assert abs(dark - bright) < 0.5 * max(dark, bright)
