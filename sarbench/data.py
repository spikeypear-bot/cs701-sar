"""The course data in TA Zichen's folder layout (see README): every image has one class label and >= 1 box.
Loads images and boxes at a fixed input size and maps predicted boxes back to original pixels. Options: input
channels from SAR filters (input_channels), and two training augmentations, BoxPaste and 90-degree rotations."""
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import tv_tensors
from torchvision.ops import box_convert
from torchvision.transforms import v2

# Mean and std of all pixels of the 9,392 train images, read as gray, resized to
# 512 x 512 and scaled to [0, 1] (computed once).
MEAN, STD = 0.2526, 0.2133
CHANNELS = ('raw', 'nlm', 'cv', 'gre')  # the choices of input_channels


def _cv2():
    """OpenCV, imported only for the filter channels, at one thread per call: every data worker calls it."""
    import cv2
    cv2.setNumThreads(1)
    return cv2


def _gre_kernels(alpha=2.0, radius=6):
    """1D kernels of 'gre': weights exp(-|d| / alpha) over the whole window, and over the side after the pixel and
    the side before it (the centre excluded), each summing to 1. The 2D weights exp(-(|dx| + |dy|) / alpha) are
    their outer product, so each half-window mean is two 1D passes. cv2.sepFilter2D correlates, so an entry right
    of the kernel centre weighs the pixel right of (or below) the output pixel."""
    d = np.arange(-radius, radius + 1)
    whole = np.exp(-np.abs(d) / alpha).astype(np.float32)
    after, before = whole * (d > 0), whole * (d < 0)
    return whole / whole.sum(), after / after.sum(), before / before.sum()


def input_channels(gray, channels):
    """gray: uint8 [H, W] -> uint8 [H, W, len(channels)], one map per name in channels:
    'raw': the image itself;
    'nlm': non-local means (Buades et al., 2005): speckle averaged over similar patches, edges kept (~80 ms);
    'cv': the local coefficient of variation (std / mean over 5 x 5, as the Lee filter measures it), high at bright
    point targets and edges, low on flat ground; [0, 2] scaled to [0, 255];
    'gre': gradient by ratio (ROEWA, Fjortoft et al., 1998, as in SAR-SIFT and SARDet-100K's filter augmentation):
    the magnitude of the log ratios of exponentially weighted means on opposite sides of each pixel, horizontally and
    vertically. A ratio, not a difference, so speckle, which is multiplicative, does not raise it on bright ground;
    [0, 2] scaled to [0, 255]."""
    maps = []
    for name in channels:
        if name == 'raw':
            maps.append(gray)
        elif name == 'nlm':
            maps.append(_cv2().fastNlMeansDenoising(np.ascontiguousarray(gray), None, h=10, templateWindowSize=7,
                                                    searchWindowSize=21))
        elif name == 'cv':
            cv2, x = _cv2(), gray.astype(np.float32)
            mean = cv2.blur(x, (5, 5))
            std = np.sqrt(np.maximum(cv2.blur(x * x, (5, 5)) - mean ** 2, 0))
            maps.append((np.clip(std / (mean + 1), 0, 2) * 127.5).astype(np.uint8))
        elif name == 'gre':
            cv2, x = _cv2(), gray.astype(np.float32) + 1  # + 1: no log of 0
            whole, after, before = _gre_kernels()
            gx = np.log(cv2.sepFilter2D(x, -1, after, whole) / cv2.sepFilter2D(x, -1, before, whole))  # right / left
            gy = np.log(cv2.sepFilter2D(x, -1, whole, after) / cv2.sepFilter2D(x, -1, whole, before))  # below / above
            maps.append((np.clip(np.hypot(gx, gy), 0, 2) * 127.5).astype(np.uint8))
        else:
            raise ValueError(f'unknown input channel {name!r}, expected one of {CHANNELS}')
    return np.stack(maps, axis=-1)


def channel_stats(root, channels, n=256):
    """[means, stds] of the input channels in [0, 1] units, for Normalize. 'raw' keeps MEAN, STD; the filter
    channels are measured on the first n training images (the split files are shuffled, see make_split.py)."""
    means, stds = [MEAN] * len(channels), [STD] * len(channels)
    if all(name == 'raw' for name in channels):
        return [means, stds]
    root = Path(root)
    with open(root / 'classification' / 'train.csv') as f:
        rows = list(csv.DictReader(f))[:n]
    total, total_sq, count = np.zeros(len(channels)), np.zeros(len(channels)), 0
    for row in rows:
        x = input_channels(np.asarray(Image.open(root / row['image']).convert('L')), channels)
        x = x.reshape(-1, len(channels)) / 255
        total, total_sq, count = total + x.sum(axis=0), total_sq + (x ** 2).sum(axis=0), count + len(x)
    for c, name in enumerate(channels):
        if name != 'raw':
            means[c] = float(total[c] / count)
            stds[c] = float(np.sqrt(total_sq[c] / count - means[c] ** 2))
    return [means, stds]


class BoxPaste:
    """Same-class BoxPaste (after Suo et al., 2022, for SAR ships): small objects are cut out by their boxes, plus a
    margin, from training images of the same class and size, and pasted at free spots, each with a new box. The image
    keeps its one class, so its classification label stays right. Each patch is scaled to the brightness of the spot
    it covers and its margin is blended in, so no hard seam gives it away."""

    def __init__(self, items, prob, max_pastes, max_size, margin=3):
        """items: (image path, (h, w), category id, [x, y, w, h]) of every box that may be copied."""
        self.pool = defaultdict(list)  # (category id, (h, w)) -> [(path, box)]: same size, so the same scale
        for path, size, category, box in items:
            if max(box[2], box[3]) <= max_size:
                self.pool[category, size].append((path, box))
        self.prob, self.max_pastes, self.margin = prob, max_pastes, margin
        ramp = lambda n: np.minimum(np.minimum(np.arange(n), np.arange(n)[::-1]) + 1, margin + 1) / (margin + 1)
        self.alpha = lambda h, w: np.minimum.outer(ramp(h), ramp(w))  # 1 on the object, fading over the margin

    def __call__(self, gray, boxes, category):
        """gray uint8 [H, W], boxes [[x, y, w, h], ...] of `category` -> (gray, boxes), maybe with pasted objects."""
        donors = self.pool.get((category, gray.shape))
        if not donors or random.random() >= self.prob:
            return gray, boxes
        gray, boxes, m = gray.astype(np.float32), list(boxes), self.margin
        height, width = gray.shape
        for _ in range(random.randint(1, self.max_pastes)):
            path, (bx, by, bw, bh) = random.choice(donors)
            x0, y0 = int(bx) - m, int(by) - m
            x1, y1 = int(np.ceil(bx + bw)) + m, int(np.ceil(by + bh)) + m
            if x0 < 0 or y0 < 0 or x1 > width or y1 > height:
                continue  # the margin would leave the donor image
            patch = np.asarray(Image.open(path).convert('L'), dtype=np.float32)[y0:y1, x0:x1]
            ph, pw = patch.shape
            for _ in range(10):  # a spot whose patch overlaps no box, pasted ones included
                px, py = random.randint(0, width - pw), random.randint(0, height - ph)
                if not any(px < x + w and x < px + pw and py < y + h and y < py + ph for x, y, w, h in boxes):
                    break
            else:
                continue
            spot = gray[py:py + ph, px:px + pw]
            ring = np.ones_like(patch, dtype=bool)
            ring[m:-m, m:-m] = False  # the margin: the donor's background around the object
            scale = np.clip((spot.mean() + 1) / (patch[ring].mean() + 1), 0.5, 2)
            alpha = self.alpha(ph, pw)
            gray[py:py + ph, px:px + pw] = alpha * patch * scale + (1 - alpha) * spot
            boxes.append([px + bx - x0, py + by - y0, bw, bh])
        return gray.clip(0, 255).astype(np.uint8), boxes


def rot90(image, boxes, k):
    """Rotate a square image [C, S, S] and its XYXY boxes by k x 90 degrees; boxes stay exact at right angles."""
    size = image.shape[-1]
    for _ in range(k):
        x1, y1, x2, y2 = boxes.unbind(dim=-1)
        boxes = torch.stack([y1, size - x2, y2, size - x1], dim=-1)  # torch.rot90: pixel (x, y) -> (y, S - 1 - x)
    return torch.rot90(image, k, dims=(-2, -1)), boxes


class SARMultiTask(Dataset):
    """The images of classification/{split}.csv with their boxes from detection/instances_{split}.json.
    An item is (image [C, size, size], label 0..8, target), target = {'boxes': XYXY on the resized
    image, 'labels': 1..9, 'image_id': COCO id, 'orig_size': (h, w)}. C = len(channels), see input_channels.
    Training only: boxpaste = BoxPaste options {prob, max_pastes, max_size, classes (names)}, rot90."""

    def __init__(self, root, split, train=False, size=512, limit=None, channels=('raw',), stats=None,
                 boxpaste=None, rot90=False):
        self.root = Path(root)
        with open(self.root / 'classification' / f'{split}.csv') as f:
            self.rows = list(csv.DictReader(f))[:limit]
        coco = json.loads((self.root / 'detection' / f'instances_{split}.json').read_text())
        self.coco_images = {img['file_name']: img for img in coco['images']}  # key = CSV 'image'
        self.annotations = defaultdict(list)  # COCO image id -> its boxes
        for ann in coco['annotations']:
            self.annotations[ann['image_id']].append(ann)
        self.channels = tuple(channels)
        means, stds = stats or channel_stats(root, self.channels)
        self.boxpaste = None
        if train and boxpaste:
            ids = {c['name']: c['id'] for c in coco['categories']}
            classes = {ids[name] for name in boxpaste['classes']}
            items = []  # the boxes of these classes in this split's images (the training set: no holdout leaks)
            for row in self.rows:
                info = self.coco_images[row['image']]
                items += [(self.root / row['image'], (info['height'], info['width']), ann['category_id'], ann['bbox'])
                          for ann in self.annotations[info['id']] if ann['category_id'] in classes]
            self.boxpaste = BoxPaste(items, boxpaste['prob'], boxpaste['max_pastes'], boxpaste['max_size'])
        self.rot90 = train and rot90
        flips = [v2.RandomHorizontalFlip(), v2.RandomVerticalFlip()] if train else []
        self.transform = v2.Compose([
            v2.ToImage(),
            v2.ConvertBoundingBoxFormat('XYXY'),  # COCO [x, y, w, h] -> [x1, y1, x2, y2]
            v2.Resize((size, size)),  # stretches non-square images; boxes are scaled with the pixels
            *flips,  # overhead imagery has no canonical "up"
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(means, stds),
        ])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        info = self.coco_images[row['image']]
        anns = self.annotations[info['id']]
        orig_size = (info['height'], info['width'])
        gray = np.asarray(Image.open(self.root / row['image']).convert('L'))  # PIL: torchvision.io cannot read .bmp
        xywh, labels = [a['bbox'] for a in anns], [a['category_id'] for a in anns]
        if self.boxpaste:  # before the filters, so pasted objects are filtered like the rest of the image
            gray, xywh = self.boxpaste(gray, xywh, labels[0])
            labels += [labels[0]] * (len(xywh) - len(labels))
        boxes = tv_tensors.BoundingBoxes(xywh, format='XYWH', canvas_size=orig_size,
                                         dtype=torch.float32)  # integer boxes would be truncated by Resize
        image, boxes = self.transform(input_channels(gray, self.channels), boxes)
        if self.rot90:
            image, rotated = rot90(image, boxes, int(torch.randint(4, ())))
            boxes = tv_tensors.wrap(rotated, like=boxes)
        target = {'boxes': boxes, 'labels': torch.tensor(labels), 'image_id': info['id'], 'orig_size': orig_size}
        return image, int(row['label']), target


class SARImages(Dataset):
    """The images listed in a COCO-style images.json (course val and test: no labels), preprocessed as
    SARMultiTask does without augmentation. An item is (image [C, size, size], its entry of images.json)."""

    def __init__(self, images_json, root, size=512, channels=('raw',), stats=None):
        self.root = Path(root)  # the file names in images.json are relative to it
        self.images = json.loads(Path(images_json).read_text())['images']
        self.channels = tuple(channels)
        if stats is None and self.channels != ('raw',) * len(self.channels):
            raise ValueError('filter channels need the training stats of the run (metrics.json: channel_stats)')
        means, stds = stats or [[MEAN] * len(self.channels), [STD] * len(self.channels)]
        self.transform = v2.Compose([
            v2.ToImage(),
            v2.Resize((size, size)),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(means, stds),
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        info = self.images[i]
        gray = np.asarray(Image.open(self.root / info['file_name']).convert('L'))
        return self.transform(input_channels(gray, self.channels)), info


def collate_images(batch):
    """Stack the images of SARImages; their images.json entries stay a list."""
    images, infos = zip(*batch)
    return torch.stack(images), list(infos)


def collate(batch):
    """Stack images and labels; targets stay a list because images have different numbers of boxes."""
    images, labels, targets = zip(*batch)
    return torch.stack(images), torch.tensor(labels), list(targets)


def to_original_xywh(boxes_xyxy, orig_size, size=512):
    """Undo the resize: XYXY boxes on the size x size input -> COCO [x, y, w, h] in original pixels."""
    h, w = orig_size
    return box_convert(boxes_xyxy * boxes_xyxy.new_tensor([w, h, w, h]) / size, 'xyxy', 'xywh')
