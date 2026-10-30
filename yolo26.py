"""YOLO26 (default YOLO26x-P2) trained by Ultralytics' own trainer, as a groupmate's run (Codabench Δm +41.16):
COCO weights where the shapes fit, chosen layers frozen to stay under the 40 M trainable limit, optional
Albumentations (multiplicative noise, gamma, brightness/contrast) and any other Ultralytics training argument via
--hyp. Detection only: the image class is the class of the image's highest-scored box (every image holds one class).
Needs a newer Ultralytics than train.py's 8.3.100, so it runs in its own conda env (train.slurm picks cs701y26 for
this script; setup in docs/claude.md). Examples:
  sbatch --mem=64GB --time=1-00:00:00 train.slurm yolo26.py --freeze 5-8 --epochs 100 \\
      --mult-noise 0.1 --gamma 0.2 --brightness-contrast 0.2 --name y26x-p2-f5-8-e100
  python yolo26.py --predict runs/<run> cs701-sar-course-data/val/images.json   # -> <run>/submission_val.zip
  python yolo26.py --score runs/<run> --head one2one   # holdout scores with the other head, nothing written
Writes runs/<name>_<time>/: Ultralytics' folder (ultralytics/, weights/last.pt saved every epoch), args.json,
predictions_test.json and metrics.json (the holdout, scored as train.py scores it), and a row in runs/results.csv.
--stop-hours (set by train.slurm from the job's time limit): after an epoch, exit with code 75 if the next one and the
final scoring might not fit; train.slurm then requeues the job and the requeued job continues with --resume."""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import yaml

from make_submission import write_submission
from sarbench.metrics import classification_metrics, detection_metrics
from train import MAX_PARAMETERS, MAX_TRAINABLE, RESUME_EXIT, append_results, parse_blocks, save_code_version
from ultralytics_baseline import IGNORE_CLASS, write_yolo_dataset


def parse_hyp(spec):
    """'mosaic=0,scale=0.2,cos_lr=true' -> {'mosaic': 0, 'scale': 0.2, 'cos_lr': True} (values read as YAML)"""
    hyp = {}
    for part in filter(None, spec.split(',')):
        name, _, value = part.partition('=')
        hyp[name.strip()] = yaml.safe_load(value)
    return hyp


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data', type=Path, default=Path('data'), help='the course data as train.py reads it')
    p.add_argument('--model', default='yolo26x-p2',
                   help='an Ultralytics model name; -p2: its yaml, with the plain model\'s COCO weights where the '
                        'shapes fit (default: yolo26x-p2)')
    p.add_argument('--freeze', type=parse_blocks, default=[],
                   help='layers kept at their COCO weights, e.g. 5-8 or 5,6,7,8 (Ultralytics\' freeze list)')
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--imgsz', type=int, default=512)
    p.add_argument('--batch', type=int, default=16)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--mult-noise', type=float, default=0.0,
                   help='Albumentations MultiplicativeNoise, multiplier uniform in 1 +- this (e.g. 0.1; 0: off)')
    p.add_argument('--elementwise', action='store_true',
                   help='--mult-noise per pixel (speckle-like) instead of one multiplier per image (a brightness '
                        'change; Albumentations\' default)')
    p.add_argument('--gamma', type=float, default=0.0,
                   help='Albumentations RandomGamma, gamma uniform in 1 +- this (e.g. 0.2 = its default 80-120; 0: off)')
    p.add_argument('--brightness-contrast', type=float, default=0.0,
                   help='Albumentations RandomBrightnessContrast, both limits (e.g. 0.2, its default; 0: off)')
    p.add_argument('--jpeg', type=int, default=0, metavar='Q',
                   help='Albumentations ImageCompression, JPEG quality uniform in Q-100 (e.g. 50; 0: off): PNGs '
                        'come from a second source in only four classes (finding 21)')
    p.add_argument('--albu-p', type=float, default=0.5, help='probability of each Albumentations transform (default: 0.5)')
    p.add_argument('--class-scale', type=parse_hyp, default={}, metavar='CLASS=A,...',
                   help='scale augmentation per class instead of Ultralytics\' one factor per mosaic: each training '
                        'image resized by 1 +- A (uniform) before Mosaic, A by its class, default=A for the rest, '
                        'e.g. default=0.25,ship=0.5,oil_tank=0.1; sets Ultralytics\' scale to 0')
    p.add_argument('--hyp', type=parse_hyp, default={},
                   help='more Ultralytics training arguments, e.g. mosaic=0,scale=0.2,flipud=0.5,optimizer=AdamW')
    p.add_argument('--o2m-final', type=float,
                   help='ProgLoss: final weight of the one-to-many loss, the only loss reaching the backbone and neck '
                        '(Ultralytics: 0.8 decaying to 0.1; 0.8 = no decay; default: Ultralytics\')')
    p.add_argument('--stal-min', type=float,
                   help='STAL: ground-truth sides below this (input px) are enlarged to it when picking candidate '
                        'anchors (Ultralytics: the second stride, 8 px with P2; default: Ultralytics\')')
    p.add_argument('--ignore-dim', type=lambda s: s.split(','), default=[], metavar='CLASSES',
                   help='train boxes of these classes (e.g. bridge,playground) with no bright return (brightest '
                        'pixel at most the image\'s 90th percentile; finding 21) give no loss: not a target, and the '
                        'anchors inside them are not trained as background either (default: none)')
    p.add_argument('--cls-weights', type=parse_hyp, default={}, metavar='CLASS=W,...',
                   help='weight of these classes\' columns in the box classification loss (BCE, both heads; '
                        'Ultralytics\' class_weights), the rest 1, e.g. bridge=2,playground=2 (default: none)')
    p.add_argument('--name', help='run name; the run folder is <out>/<name>_<time> (needed for training)')
    p.add_argument('--out', type=Path, default=Path('runs'))
    p.add_argument('--stop-hours', type=float, help='see the description; train.slurm sets it')
    p.add_argument('--resume', type=Path, help='continue the run in this folder from its weights/last.pt')
    p.add_argument('--predict', type=Path, nargs=2, metavar=('RUN', 'IMAGES_JSON'),
                   help='write the submission zip of a finished run for images.json (no training)')
    p.add_argument('--root', type=Path, help='--predict: the folder the file names in images.json are relative to '
                                             '(default: the folder of images.json)')
    p.add_argument('--zip', type=Path, help='--predict: the zip (default: <run>/submission_<folder of images.json>.zip)')
    p.add_argument('--score', type=Path, metavar='RUN', help='score a finished run on the holdout (no training)')
    p.add_argument('--head', choices=['one2many', 'one2one'], default='one2many',
                   help='--predict / --score: YOLO26\'s one-to-many head + NMS (Ultralytics\' default, used for every '
                        'run so far) or its NMS-free one-to-one head')
    args = p.parse_args()
    if not (args.predict or args.score or args.resume or args.name):
        p.error('training needs --name')
    if args.class_scale and 'scale' in args.hyp:
        p.error('--class-scale replaces --hyp scale')
    return args


def albumentations(args):
    """The Albumentations transforms asked for, or None (Ultralytics then uses its own default set)."""
    import albumentations as A
    transforms = []
    if args.mult_noise:
        transforms.append(A.MultiplicativeNoise(multiplier=(1 - args.mult_noise, 1 + args.mult_noise),
                                                per_channel=False, elementwise=args.elementwise, p=args.albu_p))
    if args.gamma:
        transforms.append(A.RandomGamma(gamma_limit=(100 * (1 - args.gamma), 100 * (1 + args.gamma)), p=args.albu_p))
    if args.brightness_contrast:
        transforms.append(A.RandomBrightnessContrast(brightness_limit=args.brightness_contrast,
                                                     contrast_limit=args.brightness_contrast, p=args.albu_p))
    if getattr(args, 'jpeg', 0):  # runs before the flag: no key
        transforms.append(A.ImageCompression(quality_range=(args.jpeg, 100), p=args.albu_p))
    return transforms or None


def patch_class_scale(scales, default):
    """--class-scale: training images (Ultralytics' augment=True dataset) resized by 1 +- a, a = scales[class index]
    or default, before Mosaic places them. Ultralytics keeps YOLO labels normalised until Mosaic, which denormalises
    them by the returned shape, so the boxes follow the image."""
    import random
    import cv2
    from ultralytics.data.base import BaseDataset
    load = BaseDataset.load_image

    def scaled(self, i, *args, **kwargs):
        im, hw0, hw = load(self, i, *args, **kwargs)
        cls = self.labels[i]['cls']
        # one class per image; min() skips --ignore-dim's IGNORE_CLASS, the highest index
        a = scales.get(int(cls.min()), default) if len(cls) else default
        if not self.augment or not a:
            return im, hw0, hw
        s = random.uniform(1 - a, 1 + a)
        im = cv2.resize(im, (max(1, round(hw[1] * s)), max(1, round(hw[0] * s))), interpolation=cv2.INTER_LINEAR)
        return im, hw0, im.shape[:2]
    BaseDataset.load_image = scaled


def dim_boxes(root, classes):
    """--ignore-dim: ids of the train boxes of these classes whose brightest pixel is at most their image's 90th
    percentile (no bright return)."""
    import numpy as np
    from PIL import Image
    coco = json.loads((root / 'detection' / 'instances_train.json').read_text())
    ids = {c['name']: c['id'] for c in coco['categories']}
    if unknown := set(classes) - set(ids):
        sys.exit(f'--ignore-dim: unknown classes {sorted(unknown)}')
    wanted, boxes = {ids[c] for c in classes}, {}
    for a in coco['annotations']:
        if a['category_id'] in wanted:
            boxes.setdefault(a['image_id'], []).append(a)
    dim = set()
    for im in coco['images']:
        if im['id'] in boxes:
            pixels = np.asarray(Image.open(root / im['file_name']).convert('L'))
            p90 = np.percentile(pixels, 90, method='inverted_cdf')
            for a in boxes[im['id']]:
                x, y, w, h = a['bbox']
                if pixels[int(y):int(y + h + 1), int(x):int(x + w + 1)].max() <= p90:
                    dim.add(a['id'])
    total = sum(map(len, boxes.values()))
    print(f'--ignore-dim {",".join(classes)}: {len(dim)} of {total} train boxes ignored', flush=True)
    return dim


def patch_losses(o2m_final=None, stal_min=None, ignore_cls=None):
    """--o2m-final, --stal-min: YOLO26 loss constants Ultralytics hard-codes (utils/loss.py E2ELoss, utils/tal.py
    TaskAlignedAssigner), set after their own __init__. None keeps Ultralytics' value. ignore_cls (--ignore-dim):
    boxes of this label class are dropped from the targets, and anchors whose centre lies inside one (sides enlarged
    to STAL's floor, as the assigner does) get no classification loss unless another box assigns them."""
    import torch
    from ultralytics.utils import loss, tal

    def after_init(cls, name, value):
        init = cls.__init__

        def patched(self, *a, **k):
            init(self, *a, **k)
            setattr(self, name, value)
        cls.__init__ = patched

    if o2m_final is not None:
        after_init(loss.E2ELoss, 'final_o2m', o2m_final)
    if stal_min is not None:
        after_init(tal.TaskAlignedAssigner, 'stride_val', stal_min)
    if ignore_cls is not None:
        assigned = loss.v8DetectionLoss.get_assigned_targets_and_loss

        def without_ignored(self, preds, batch, **kwargs):
            ignored = batch['cls'].view(-1) == ignore_cls
            if not ignored.any():
                return assigned(self, preds, batch, **kwargs)
            anchors, strides = tal.make_anchors(preds['feats'], self.stride, 0.5)
            imgsz = torch.tensor(preds['feats'][0].shape[2:], device=self.device) * self.stride[0]  # h, w
            xywh = batch['bboxes'][ignored].to(self.device) * imgsz[[1, 0, 1, 0]]
            xywh[:, 2:] = xywh[:, 2:].clamp(min=self.assigner.stride_val)
            points = (anchors * strides)[None]  # (1, anchors, 2) in input px
            inside = ((points > xywh[:, None, :2] - xywh[:, None, 2:] / 2) &
                      (points < xywh[:, None, :2] + xywh[:, None, 2:] / 2)).all(-1).float()  # (ignored, anchors)
            covered = torch.zeros(preds['scores'].shape[0], inside.shape[1], device=self.device).index_add_(
                0, batch['batch_idx'][ignored].long().to(self.device), inside) > 0  # (images, anchors)
            bce = self.bce
            self.bce = lambda p, t: bce(p, t) * ~(covered[..., None] & (t == 0).all(-1, keepdim=True))
            kept = {**batch, **{k: batch[k][~ignored] for k in ('cls', 'bboxes', 'batch_idx')}}
            try:
                return assigned(self, preds, kept, **kwargs)
            finally:
                self.bce = bce
        loss.v8DetectionLoss.get_assigned_targets_and_loss = without_ignored


def set_class_weights(model, weights):
    """--cls-weights: Ultralytics' own model.class_weights (v8DetectionLoss multiplies each class column of its BCE by
    them), set at the end of the trainer's setup; the loss reads them when it is built, at the first training step."""
    import torch
    from ultralytics.utils.torch_utils import unwrap_model

    def set_weights(trainer):
        unwrap_model(trainer.model).class_weights = torch.tensor(weights, device=trainer.device)
        print(f'--cls-weights: class weights {weights}', flush=True)
    model.add_callback('on_pretrain_routine_end', set_weights)


def callbacks(model, run_dir, args, process_start, counts):
    """Checks the trainable count once the trainer has frozen its layers (aborts over the limit), and --stop-hours
    after each saved epoch (weights/last.pt is written before on_model_save). counts['train_start'] (set before
    training) and train_seconds.txt add up the training time of every job of the run."""
    epoch_ends, longest = [process_start], [0.0]

    def check_limits(trainer):
        counts['parameters'] = sum(q.numel() for q in trainer.model.parameters())
        counts['trainable'] = sum(q.numel() for q in trainer.model.parameters() if q.requires_grad)
        print(f'parameters: {counts["parameters"]:,} total, {counts["trainable"]:,} trainable', flush=True)
        if counts['parameters'] > MAX_PARAMETERS or counts['trainable'] > MAX_TRAINABLE:
            sys.exit(f'over the limits ({MAX_PARAMETERS / 1e6:.0f} M total, {MAX_TRAINABLE / 1e6:.0f} M trainable): '
                     'freeze more layers')
        found = sum(int((lb['cls'] == IGNORE_CLASS).sum()) for lb in trainer.train_loader.dataset.labels)
        if found != counts['ignored']:  # Ultralytics silently drops an image whose labels fail its checks
            sys.exit(f'--ignore-dim: {found} ignored boxes in the train labels, {counts["ignored"]} written')

    def maybe_stop(trainer):
        now = time.time()
        longest[0] = max(longest[0], now - epoch_ends[-1])
        epoch_ends.append(now)
        if args.stop_hours is None or trainer.epoch + 1 >= trainer.epochs:
            return
        if now - process_start + 2 * longest[0] > args.stop_hours * 3600:  # the next epoch, then the scoring
            print(f'stopped after epoch {trainer.epoch + 1}: the next might end past --stop-hours {args.stop_hours}; '
                  f'continue with --resume {run_dir}', flush=True)
            (run_dir / 'train_seconds.txt').write_text(str(earlier_seconds(run_dir) + now - counts['train_start']))
            if os.environ.get('RESUME_NOTE'):  # set by train.slurm, which requeues the job
                Path(os.environ['RESUME_NOTE']).write_text(str(run_dir))
            sys.exit(RESUME_EXIT)

    model.add_callback('on_pretrain_routine_end', check_limits)
    model.add_callback('on_model_save', maybe_stop)


def earlier_seconds(run_dir):
    """Training seconds of the run's earlier jobs (0 before its first stop)."""
    path = run_dir / 'train_seconds.txt'
    return float(path.read_text()) if path.exists() else 0.0


def predict(weights, images, root, size, batch, head='one2many'):
    """Each image's boxes ([x1, y1, x2, y2] on the size x size input, as write_submission takes them), classes 0..8
    and scores, at confidence >= 0.001, at most 100 per image; the image's class is its highest-scored box's
    (class 0 if it has none). head: one2many = NMS on the one-to-many head; one2one = the NMS-free head
    (Ultralytics' nms=False)."""
    from ultralytics import YOLO
    model = YOLO(str(weights))
    preds = {}
    for start in range(0, len(images), batch):
        chunk = images[start:start + batch]
        for im, r in zip(chunk, model.predict([str(root / im['file_name']) for im in chunk], imgsz=size, conf=0.001,
                                              iou=0.7, max_det=100, batch=batch, verbose=False,
                                              nms=False if head == 'one2one' else None)):
            scale = [size / im['width'], size / im['height']] * 2  # original pixels -> input pixels
            boxes = [[v * f for v, f in zip(box, scale)] for box in r.boxes.xyxy.tolist()]
            classes, scores = r.boxes.cls.int().tolist(), r.boxes.conf.tolist()
            label = classes[max(range(len(scores)), key=scores.__getitem__)] if scores else 0
            preds[im['file_name']] = (label, boxes, classes, scores)
    return preds


def score_holdout(weights, root, size, batch, head='one2many'):
    """The holdout's classification and detection scores (train.py's metrics) and its detections."""
    gt_json = root / 'detection' / 'instances_test.json'
    gt = json.loads(gt_json.read_text())
    preds = predict(weights, gt['images'], root, size, batch, head)
    truth = {a['image_id']: a['category_id'] - 1 for a in gt['annotations']}
    detections = []
    for im in gt['images']:
        _, boxes, classes, scores = preds[im['file_name']]
        sx, sy = im['width'] / size, im['height'] / size
        detections += [{'image_id': im['id'], 'category_id': c + 1, 'score': s,
                        'bbox': [x1 * sx, y1 * sy, (x2 - x1) * sx, (y2 - y1) * sy]}
                       for (x1, y1, x2, y2), c, s in zip(boxes, classes, scores)]
    ids = [im['id'] for im in gt['images']]
    metrics = {**classification_metrics([truth[i] for i in ids], [preds[im['file_name']][0] for im in gt['images']]),
               **detection_metrics(gt_json, detections, ids)}
    return metrics, detections


def main():
    process_start = time.time()
    args = parse_args()
    if args.predict:
        run, images_json = args.predict
        options = json.loads((run / 'args.json').read_text())
        images = json.loads(images_json.read_text())['images']
        zip_path = args.zip or run / f'submission_{images_json.parent.name}.zip'
        write_submission(images, predict(run / 'ultralytics' / 'weights' / 'last.pt', images,
                                         args.root or images_json.parent, options['imgsz'], options['batch'],
                                         args.head),
                         size=options['imgsz'], out=zip_path)
        print(f'{len(images)} images -> {zip_path}')
        return
    if args.score:
        options = json.loads((args.score / 'args.json').read_text())
        test, _ = score_holdout(args.score / 'ultralytics' / 'weights' / 'last.pt', Path(options['data']),
                                options['imgsz'], options['batch'], args.head)
        print(f'{args.score.name} --head {args.head}: macro_f1 {test["macro_f1"]:.4f} mAP {test["mAP"]:.4f} '
              f'AP50 {test["AP50"]:.4f}')
        return

    from ultralytics import YOLO
    if args.resume:  # the run's own options, except --stop-hours (this job's)
        run_dir, stop_hours = args.resume, args.stop_hours
        args = argparse.Namespace(**{**json.loads((run_dir / 'args.json').read_text()), 'resume': run_dir,
                                     'stop_hours': stop_hours})
        args.data, args.out, args.freeze = Path(args.data), Path(args.out), list(args.freeze)
        commit, dirty = args.code
        model = YOLO(str(run_dir / 'ultralytics' / 'weights' / 'last.pt'))
    else:
        run_dir = args.out / f'{args.name}_{time.strftime("%Y%m%d-%H%M%S")}'
        run_dir.mkdir(parents=True)
        commit, dirty = save_code_version(run_dir)
        args.code = [commit, dirty]
        (run_dir / 'args.json').write_text(json.dumps(vars(args), indent=2, default=str))
        plain = args.model.removesuffix('-p2')
        model = YOLO(f'{args.model}.yaml').load(f'{plain}.pt') if args.model.endswith('-p2') else YOLO(f'{plain}.pt')
    counts = {}
    callbacks(model, run_dir, args, process_start, counts)
    ignore_dim = getattr(args, 'ignore_dim', [])  # runs before these flags: None / []
    patch_losses(getattr(args, 'o2m_final', None), getattr(args, 'stal_min', None),
                 IGNORE_CLASS if ignore_dim else None)
    ignore = dim_boxes(args.data, ignore_dim) if ignore_dim else set()
    counts['ignored'] = len(ignore)
    data_yaml = write_yolo_dataset(args.data, ignore)
    index = {name: int(i) for i, name in yaml.safe_load(data_yaml.read_text())['names'].items()}
    if cls_weights := dict(getattr(args, 'cls_weights', None) or {}):  # runs before the flag: no key
        if unknown := set(cls_weights) - set(index):
            sys.exit(f'--cls-weights: unknown classes {sorted(unknown)}')
        set_class_weights(model, [cls_weights.get(name, 1.0) for name in sorted(index, key=index.get)])
    if class_scale := dict(getattr(args, 'class_scale', None) or {}):  # runs before the flag: no key
        default = class_scale.pop('default', 0.5)  # Ultralytics' scale
        if unknown := set(class_scale) - set(index):
            sys.exit(f'--class-scale: unknown classes {sorted(unknown)}')
        patch_class_scale({index[name]: a for name, a in class_scale.items()}, default)
        args.hyp = {**args.hyp, 'scale': 0.0}
    counts['train_start'] = time.time()
    common = dict(augmentations=albumentations(args), workers=args.workers)  # also taken on resume
    if args.resume:
        model.train(resume=True, **common)
    else:
        model.train(**{**dict(data=str(data_yaml), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
                              seed=args.seed, freeze=args.freeze or None, project=str(run_dir.absolute()),
                              name='ultralytics', exist_ok=True, plots=False, **common), **args.hyp})
    train_hours = (earlier_seconds(run_dir) + time.time() - counts['train_start']) / 3600

    test, detections = score_holdout(run_dir / 'ultralytics' / 'weights' / 'last.pt', args.data, args.imgsz,
                                     args.batch)
    summary = {'run': run_dir.name, 'commit': commit, 'dirty': dirty, 'args': vars(args),
               'parameters': counts['parameters'], 'trainable_parameters': counts['trainable'],
               'train_hours': train_hours, 'test': test}
    (run_dir / 'predictions_test.json').write_text(json.dumps(detections))
    (run_dir / 'metrics.json').write_text(json.dumps(summary, indent=2, default=str))
    append_results(summary, args.out / 'results.csv')
    print(f'test: accuracy {test["accuracy"]:.4f} macro_f1 {test["macro_f1"]:.4f} mAP {test["mAP"]:.4f} '
          f'AP50 {test["AP50"]:.4f}; training {train_hours:.2f} h -> {run_dir}', flush=True)


if __name__ == '__main__':
    main()
