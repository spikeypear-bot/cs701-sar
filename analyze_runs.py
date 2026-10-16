"""Break down the holdout scores of train.py runs: AP by object size, AP and F1 per class, AP by image size.
Example: python analyze_runs.py runs/swin_b-lora_20261016-114202 runs/swin_b-moelora_20261016-171004
Reads each run's metrics.json and predictions_test.json (the holdout, see make_split.py); needs no GPU."""
import argparse
import contextlib
import io
import json
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# Box area in original pixels. COCO's "small" (< 32²) holds most boxes here, so it is split at one ViT patch (16²)
SIZES = {'tiny <16': (0, 16**2), 'small 16-32': (16**2, 32**2), 'medium 32-96': (32**2, 96**2),
         'large >96': (96**2, 1e10)}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('runs', type=Path, nargs='+', help='run folders of train.py')
    p.add_argument('--gt', type=Path, default=Path('data/detection/instances_test.json'),
                   help='the boxes of the holdout (default: the copy train.py scores as test)')
    return p.parse_args()


def mean_ap(precision):
    """COCO's mean over the precision entries that exist (-1: no ground truth boxes there); NaN if none."""
    valid = precision[precision > -1]
    return float(valid.mean()) if valid.size else float('nan')


def evaluate(gt, detections, image_ids):
    """COCOeval over `image_ids` with the SIZES area ranges; precision [IoU, recall, class, area, max dets]."""
    with contextlib.redirect_stdout(io.StringIO()):
        dt = gt.loadRes([dict(d) for d in detections]) if detections else COCO()
        ev = COCOeval(gt, dt, 'bbox')
        ev.params.imgIds = image_ids
        ev.params.areaRng = [[0, 1e10], *SIZES.values()]
        ev.params.areaRngLbl = ['all', *SIZES]
        ev.evaluate()
        ev.accumulate()
    return ev.eval['precision'][..., -1]  # at 100 detections per image, as mAP


def analyze(run, gt, names):
    """One column of the table: {row name: value}, and the run's most frequent classification mistakes."""
    summary = json.loads((run / 'metrics.json').read_text())
    detections = json.loads((run / 'predictions_test.json').read_text())
    image_ids = sorted(gt.getImgIds())
    precision = evaluate(gt, detections, image_ids)
    column = {'macro_f1': summary['test']['macro_f1'], 'mAP': mean_ap(precision[..., 0])}
    for a, size in enumerate(SIZES, 1):
        column[f'AP {size}'] = mean_ap(precision[..., a])
    for k, name in enumerate(names):
        column[f'AP {name}'] = mean_ap(precision[:, :, k, 0])
    # Classification per class from the confusion matrix (rows = truth, columns = prediction)
    cm = np.array(summary['test']['confusion_matrix'])
    tp = np.diag(cm)
    for k, name in enumerate(names):
        column[f'F1 {name}'] = 2 * tp[k] / (cm[k].sum() + cm[:, k].sum())
    # Images stored at <= 256 px are upscaled 2x to the 512 px input; the others are not
    small_images = {i for i in image_ids if max(gt.imgs[i]['width'], gt.imgs[i]['height']) <= 256}
    for label, ids in (('<=256 px', small_images), ('>256 px', set(image_ids) - small_images)):
        column[f'mAP images {label}'] = mean_ap(evaluate(gt, detections, sorted(ids))[..., 0])
    mistakes = [(cm[t, p], names[t], names[p]) for t in range(len(names)) for p in range(len(names))
                if t != p and cm[t, p]]
    return column, sorted(mistakes, reverse=True)[:5]


def main():
    args = parse_args()
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(args.gt)
    cat_ids = sorted(gt.getCatIds())
    names = [gt.cats[c]['name'] for c in cat_ids]
    columns, mistakes = zip(*(analyze(run, gt, names) for run in args.runs))

    # Ground truth counts next to each row, so thin rows (few boxes or images) are visible
    boxes = gt.loadAnns(gt.getAnnIds())
    counts = {f'AP {s}': sum(lo <= b['area'] < hi for b in boxes) for s, (lo, hi) in SIZES.items()}
    for c, name in zip(cat_ids, names):
        counts[f'AP {name}'] = len(gt.getAnnIds(catIds=[c]))
        counts[f'F1 {name}'] = len(gt.getImgIds(catIds=[c]))
    small = sum(max(i['width'], i['height']) <= 256 for i in gt.imgs.values())
    counts['mAP images <=256 px'], counts['mAP images >256 px'] = small, len(gt.imgs) - small

    headers = [run.name for run in args.runs]
    width = max(len(h) for h in headers)
    print(f'{"":24s} {"n":>6s}  ' + '  '.join(f'{h:>{width}s}' for h in headers))
    for row in columns[0]:
        n = f'{counts[row]:6d}' if row in counts else ' ' * 6
        print(f'{row:24s} {n}  ' + '  '.join(f'{100 * c[row]:>{width}.1f}' for c in columns))
    print('\nn: boxes for AP rows, images for F1 and image-size rows. Scores in %.')
    for run, top in zip(headers, mistakes):
        print(f'\n{run}, most frequent classification mistakes (truth -> predicted):')
        for n, truth, predicted in top:
            print(f'  {n:4d}  {truth} -> {predicted}')


if __name__ == '__main__':
    main()
