"""How tightly a run's boxes fit, per class, on the holdout: each ground-truth box against the prediction of its class
(score >= --score) in its image with the highest IoU; one prediction may be the best match of several boxes.
Example: python box_fit.py runs/yolov8m-e48_20261019-005152 runs/yolov8m-800-e48_20261021-002216 --classes aircraft
Columns: n = ground-truth boxes; found = share with a match at IoU >= 0.5; the rest over the found boxes only: IoU p25,
median, p75, share at IoU >= 0.75 and >= 0.9, median predicted / true width and height (> 1: too big), median centre
offset in original pixels (x, y; + = right, down) and its median length |d|; side = median true box side (sqrt of the
area) over all n. Each class also split by image size (<= 256 px: upscaled the most to the model input). Reads each
run's predictions_test.json; needs no GPU."""
import argparse
import json
from pathlib import Path

import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('runs', type=Path, nargs='+', help='run folders of train.py')
    p.add_argument('--classes', default='aircraft', help='comma-separated class names')
    p.add_argument('--score', type=float, default=0.3, help='ignore predictions below this score (default: 0.3)')
    p.add_argument('--gt', type=Path, default=Path('data/detection/instances_test.json'),
                   help='the boxes of the holdout (default: the copy train.py scores as test)')
    return p.parse_args()


def iou(box, others):
    """IoU of one xywh box with each row of others [n, 4] (xywh)."""
    x1, y1 = np.maximum(box[0], others[:, 0]), np.maximum(box[1], others[:, 1])
    x2 = np.minimum(box[0] + box[2], others[:, 0] + others[:, 2])
    y2 = np.minimum(box[1] + box[3], others[:, 1] + others[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    return inter / (box[2] * box[3] + others[:, 2] * others[:, 3] - inter)


def fit_row(pairs):
    """pairs [(true xywh, best predicted xywh or None, IoU)] -> the printed columns."""
    true = np.array([t for t, _, _ in pairs])
    side = np.median(np.sqrt(true[:, 2] * true[:, 3]))
    found = [(t, p, v) for t, p, v in pairs if v >= 0.5]
    if not found:
        return [len(pairs), 0.0] + [np.nan] * 10 + [side]
    t, p = np.array([t for t, _, _ in found]), np.array([p for _, p, _ in found])
    ious = np.array([v for _, _, v in found])
    offset = (p[:, :2] + p[:, 2:] / 2) - (t[:, :2] + t[:, 2:] / 2)  # centre of predicted - centre of true
    return [len(pairs), len(found) / len(pairs), *np.percentile(ious, [25, 50, 75]), (ious >= 0.75).mean(),
            (ious >= 0.9).mean(), np.median(p[:, 2] / t[:, 2]), np.median(p[:, 3] / t[:, 3]),
            *np.median(offset, axis=0), np.median(np.hypot(*offset.T)), side]


def main():
    args = parse_args()
    gt = json.loads(args.gt.read_text())
    ids = {c['name']: c['id'] for c in gt['categories']}
    small = {im['id'] for im in gt['images'] if max(im['width'], im['height']) <= 256}
    for run in args.runs:
        predicted = {}
        for d in json.loads((run / 'predictions_test.json').read_text()):
            if d['score'] >= args.score:
                predicted.setdefault((d['image_id'], d['category_id']), []).append(d['bbox'])
        print(f'\n{run.name} (predictions with score >= {args.score})')
        print(f'{"":21s} {"n":>4s} {"found":>5s} {"IoU25":>5s} {"IoU50":>5s} {"IoU75":>5s} {">=.75":>5s} {">=.9":>5s} '
              f'{"w x":>5s} {"h x":>5s} {"dx":>5s} {"dy":>5s} {"|d|":>5s} {"side":>5s}')
        for name in args.classes.split(','):
            pairs = []  # (image id, true xywh, best predicted xywh or None, IoU)
            for a in gt['annotations']:
                if a['category_id'] != ids[name]:
                    continue
                boxes = np.array(predicted.get((a['image_id'], a['category_id']), []), dtype=float).reshape(-1, 4)
                ious = iou(np.array(a['bbox'], dtype=float), boxes)
                best = int(ious.argmax()) if len(boxes) else None
                pairs.append((a['image_id'], a['bbox'], None if best is None else boxes[best],
                              0.0 if best is None else ious[best]))
            for label, keep in (('all', lambda i: True), ('<=256 px', lambda i: i in small),
                                ('>256 px', lambda i: i not in small)):
                group = [(t, p, v) for i, t, p, v in pairs if keep(i)]
                if not group:
                    continue
                row = fit_row(group)
                print(f'{name:12s} {label:>8s} {row[0]:4d} {100 * row[1]:5.1f} '
                      + ' '.join(f'{v:5.2f}' for v in row[2:5]) + ' '
                      + ' '.join(f'{100 * v:5.1f}' for v in row[5:7]) + ' '
                      + ' '.join(f'{v:5.2f}' for v in row[7:9]) + ' '
                      + ' '.join(f'{v:5.1f}' for v in row[9:13]))


if __name__ == '__main__':
    main()
