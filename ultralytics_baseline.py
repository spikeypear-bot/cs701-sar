"""Detection-only baseline: the same COCO YOLO trained by Ultralytics' own trainer and default recipe instead of
train.py's loop, to measure how much the recipe is worth. Ultralytics' defaults that train.py lacks: weight EMA, mosaic
(off for the last 10 epochs), scale ±50 % and translate ±10 %, HSV-value jitter, 3 warmup epochs, linear lr decay to
1 %, an effective batch of 64 (gradient accumulation), optimizer 'auto' (here AdamW, lr 0.002 x 5 / (4 + 9) =
0.00077: fewer than 10,000 iterations of 64 images). Left-right flips only (train.py also flips up-down).
Same split (train; the holdout as Ultralytics' val), same input size (images stretched to it, as Ultralytics trains),
COCO weights (3-channel: Ultralytics reads the gray images as BGR). Then the holdout is predicted as sarbench/yolo.py
does (stretched to size x size, confidence >= 0.001, NMS IoU 0.7, multi-label, 100 per image) and scored with the
same pycocotools metrics, so its mAP compares with a train.py run's test mAP. No image-class head: detection only.
Example: sbatch train.slurm ultralytics_baseline.py --model yolov8m --epochs 48 --name ul-yolov8m-e48
Writes runs/<name>_<time>/: Ultralytics' own folder (ultralytics/, weights/last.pt), predictions_test.json and
metrics.json (detection only: class_errors.py and box_fit.py read the run, analyze_runs.py does not), and YOLO labels
data/labels/*.txt plus data/yolo/{train,val}.txt and data.yaml (rewritten on every run)."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torchvision.ops import box_convert, clip_boxes_to_image

from sarbench.metrics import detection_metrics
from sarbench.yolo import WEIGHTS_URL, YOLO_MODELS
from train import save_code_version


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, default=Path('data'), help='the course data as train.py reads it')
    p.add_argument('--model', choices=YOLO_MODELS, default='yolov8m',
                   help='-p2 models: their yaml, with the plain model\'s COCO weights where the shapes fit')
    p.add_argument('--epochs', type=int, default=48)
    p.add_argument('--imgsz', type=int, default=512)
    p.add_argument('--batch', type=int, default=16)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--name', required=True)
    p.add_argument('--out', type=Path, default=Path('runs'))
    return p.parse_args()


def write_yolo_dataset(root):
    """Ultralytics' format: data/labels/<stem>.txt next to data/images/ (one 'class cx cy w h' line per box,
    normalized; boxes clipped to the image, as Ultralytics drops an image with a coordinate outside 0-1), image lists
    data/yolo/{train,val}.txt (val = the holdout) and data/yolo/data.yaml -> its path."""
    (root / 'labels').mkdir(exist_ok=True)
    (root / 'yolo').mkdir(exist_ok=True)
    for split in ('train', 'val'):
        coco = json.loads((root / 'detection' / f'instances_{split}.json').read_text())
        boxes = {}
        for a in coco['annotations']:
            boxes.setdefault(a['image_id'], []).append(a)
        paths = []
        for im in coco['images']:
            w, h = im['width'], im['height']
            lines = []
            for a in boxes.get(im['id'], []):
                x, y, bw, bh = a['bbox']
                x1, y1, x2, y2 = max(x, 0), max(y, 0), min(x + bw, w), min(y + bh, h)
                if x2 > x1 and y2 > y1:
                    lines.append(f'{a["category_id"] - 1} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} '
                                 f'{(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}')
            # not resolve(): Ultralytics finds the labels by the '/images/' in the path, which a symlink could hide
            image = (root / im['file_name']).absolute()  # data/images/00001.jpg -> labels at data/labels/00001.txt
            (root / 'labels' / f'{image.stem}.txt').write_text('\n'.join(lines) + '\n')
            paths.append(str(image))
        (root / 'yolo' / f'{split}.txt').write_text('\n'.join(paths) + '\n')
    names = {c['id'] - 1: c['name'] for c in coco['categories']}
    yaml = root / 'yolo' / 'data.yaml'
    yaml.write_text(f'path: {(root / "yolo").absolute()}\ntrain: {(root / "yolo" / "train.txt").absolute()}\n'
                    f'val: {(root / "yolo" / "val.txt").absolute()}\nnames: {json.dumps(names)}\n')
    return yaml


def coco_weights(name):
    """The COCO weights of the plain model (yolov8m-p2: yolov8m.pt), downloaded once into the torch hub cache, the
    same file sarbench/yolo.py uses."""
    path = Path(torch.hub.get_dir()) / 'ultralytics' / f'{name.split("-")[0]}.pt'
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.hub.download_url_to_file(WEIGHTS_URL.format(name.split('-')[0]), str(path))
    return path


@torch.no_grad()
def predict_holdout(weights, root, size, batch):
    """The holdout's detections in COCO result format (original pixels), as sarbench/yolo.py makes them: each image
    stretched to size x size, RGB / 255, NMS at confidence 0.001 and IoU 0.7, multi-label, at most 100 per image."""
    import cv2
    from ultralytics import YOLO
    from ultralytics.utils.ops import non_max_suppression
    model = YOLO(str(weights)).model.float().cuda().eval()
    images = json.loads((root / 'detection' / 'instances_test.json').read_text())['images']
    detections = []
    for start in range(0, len(images), batch):
        chunk = images[start:start + batch]
        pixels = [cv2.resize(cv2.imread(str(root / im['file_name'])), (size, size), interpolation=cv2.INTER_LINEAR)
                  for im in chunk]  # BGR, as Ultralytics reads them; flipped to RGB below, as it trains
        x = torch.from_numpy(np.stack([p[:, :, ::-1] for p in pixels])).permute(0, 3, 1, 2).float().cuda() / 255
        kept = non_max_suppression(model(x)[0], 0.001, 0.7, multi_label=True, max_det=100, nc=model.model[-1].nc,
                                   max_time_img=1e6)  # [n, 6]: xyxy on the input, score, class 0..8
        for im, d in zip(chunk, kept):
            boxes = clip_boxes_to_image(d[:, :4], (size, size)) * d.new_tensor([im['width'], im['height']] * 2) / size
            detections += [{'image_id': im['id'], 'category_id': int(c) + 1, 'bbox': b, 'score': s}
                           for b, c, s in zip(box_convert(boxes, 'xyxy', 'xywh').tolist(), d[:, 5].tolist(),
                                              d[:, 4].tolist())]
    return detections, [im['id'] for im in images]


def main():
    args = parse_args()
    from ultralytics import YOLO
    run_dir = args.out / f'{args.name}_{time.strftime("%Y%m%d-%H%M%S")}'
    run_dir.mkdir(parents=True)
    commit, dirty = save_code_version(run_dir)
    yaml = write_yolo_dataset(args.data)
    weights = coco_weights(args.model)
    model = YOLO(f'{args.model}.yaml').load(str(weights)) if args.model.endswith('-p2') else YOLO(str(weights))
    start = time.time()
    model.train(data=str(yaml), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, workers=args.workers,
                seed=args.seed, project=str(run_dir.absolute()), name='ultralytics', exist_ok=True, plots=False)
    train_hours = (time.time() - start) / 3600
    last = run_dir / 'ultralytics' / 'weights' / 'last.pt'  # the last epoch, as train.py reports
    detections, image_ids = predict_holdout(last, args.data, args.imgsz, args.batch)
    test = detection_metrics(args.data / 'detection' / 'instances_test.json', detections, image_ids)
    (run_dir / 'predictions_test.json').write_text(json.dumps(detections))
    (run_dir / 'metrics.json').write_text(json.dumps(
        {'run': run_dir.name, 'commit': commit, 'dirty': dirty, 'args': vars(args), 'train_hours': train_hours,
         'test': test}, indent=2, default=str))
    print(f'test: mAP {test["mAP"]:.4f} AP50 {test["AP50"]:.4f} AP75 {test["AP75"]:.4f}; training {train_hours:.2f} h '
          f'-> {run_dir}', flush=True)


if __name__ == '__main__':
    main()
