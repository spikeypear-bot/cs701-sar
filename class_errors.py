"""Why a run's AP is low, per class, on the holdout: AP at loose to strict IoU thresholds (does it find the objects but
miss the exact box?), recall (does it find them at all?), and how much image-class filtering could give.
Example: python class_errors.py runs/yolov8m-e100_20261019-222115
Columns: AP = COCO AP (IoU 0.5:0.95); AP@t = AP at IoU t alone; R@t = recall at IoU t (100 detections per image);
AP filt = AP without the detections of a class in images of another class (the image class known for sure, so an upper
bound for filtering by the predicted class); other % = share of the class's detections with score >= 0.3 that sit in
images of another class. Reads each run's predictions_test.json; needs no GPU."""
import argparse
import contextlib
import io
import json
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from analyze_runs import evaluate, mean_ap

IOUS = (0.1, 0.3, 0.5, 0.75)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('runs', type=Path, nargs='+', help='run folders of train.py')
    p.add_argument('--gt', type=Path, default=Path('data/detection/instances_test.json'),
                   help='the boxes of the holdout (default: the copy train.py scores as test)')
    return p.parse_args()


def at_ious(gt, detections, image_ids):
    """COCOeval at the IOUS thresholds -> precision [IoU, recall, class] and recall [IoU, class], all areas, 100
    detections per image."""
    with contextlib.redirect_stdout(io.StringIO()):
        ev = COCOeval(gt, gt.loadRes([dict(d) for d in detections]), 'bbox')
        ev.params.imgIds = image_ids
        ev.params.iouThrs = np.array(IOUS)
        ev.evaluate()
        ev.accumulate()
    return ev.eval['precision'][:, :, :, 0, -1], ev.eval['recall'][:, :, 0, -1]


def main():
    args = parse_args()
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(args.gt)
    cat_ids = sorted(gt.getCatIds())
    names = [gt.cats[c]['name'] for c in cat_ids]
    image_ids = sorted(gt.getImgIds())
    image_class = {i: gt.anns[gt.getAnnIds(imgIds=[i])[0]]['category_id'] for i in image_ids}  # all boxes share it
    for run in args.runs:
        detections = json.loads((run / 'predictions_test.json').read_text())
        same = [d for d in detections if d['category_id'] == image_class[d['image_id']]]
        ap, ap_same = evaluate(gt, detections, image_ids), evaluate(gt, same, image_ids)
        precision, recall = at_ious(gt, detections, image_ids)
        print(f'\n{run.name}\n{"":13s} {"AP":>5s} ' + ' '.join(f'{f"AP@{t}":>6s}' for t in IOUS)
              + ' ' + ' '.join(f'{f"R@{t}":>5s}' for t in IOUS[::2]) + f' {"AP filt":>7s} {"other %":>7s}')
        rows = []
        for k, (c, name) in enumerate(zip(cat_ids, names)):
            confident = [d for d in detections if d['category_id'] == c and d['score'] >= 0.3]
            other = sum(image_class[d['image_id']] != c for d in confident) / max(len(confident), 1)
            rows.append([mean_ap(ap[:, :, k, 0]), *(mean_ap(precision[t, :, k]) for t in range(len(IOUS))),
                         *(recall[t, k] for t in range(0, len(IOUS), 2)), mean_ap(ap_same[:, :, k, 0]), other])
        for name, row in [*zip(names, rows), ('mean', np.mean(rows, axis=0))]:
            print(f'{name:13s} {100 * row[0]:5.1f} ' + ' '.join(f'{100 * v:6.1f}' for v in row[1:5]) + ' '
                  + ' '.join(f'{100 * v:5.1f}' for v in row[5:7]) + f' {100 * row[7]:7.1f} {100 * row[8]:7.1f}')


if __name__ == '__main__':
    main()
