"""YOLO26 (default YOLO26x-P2) trained by Ultralytics' own trainer, as a groupmate's run (Codabench Δm +41.16):
COCO weights where the shapes fit, chosen layers frozen to stay under the 40 M trainable limit, optional
Albumentations (multiplicative noise, gamma, brightness/contrast) and any other Ultralytics training argument via
--hyp. Detection only: the image class is the class of the image's highest-scored box (every image holds one class).
Needs a newer Ultralytics than train.py's 8.3.100, so it runs in its own conda env (train.slurm picks cs701y26 for
this script; setup in docs/claude.md). Examples:
  sbatch --mem=64GB --time=1-00:00:00 train.slurm yolo26.py --freeze 5-8 --epochs 100 \\
      --mult-noise 0.1 --gamma 0.2 --brightness-contrast 0.2 --name y26x-p2-f5-8-e100
  python yolo26.py --predict runs/<run> cs701-sar-course-data/val/images.json   # -> <run>/submission_val.zip
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
from ultralytics_baseline import write_yolo_dataset


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
    p.add_argument('--albu-p', type=float, default=0.5, help='probability of each Albumentations transform (default: 0.5)')
    p.add_argument('--hyp', type=parse_hyp, default={},
                   help='more Ultralytics training arguments, e.g. mosaic=0,scale=0.2,flipud=0.5,optimizer=AdamW')
    p.add_argument('--name', help='run name; the run folder is <out>/<name>_<time> (needed for training)')
    p.add_argument('--out', type=Path, default=Path('runs'))
    p.add_argument('--stop-hours', type=float, help='see the description; train.slurm sets it')
    p.add_argument('--resume', type=Path, help='continue the run in this folder from its weights/last.pt')
    p.add_argument('--predict', type=Path, nargs=2, metavar=('RUN', 'IMAGES_JSON'),
                   help='write the submission zip of a finished run for images.json (no training)')
    p.add_argument('--root', type=Path, help='--predict: the folder the file names in images.json are relative to '
                                             '(default: the folder of images.json)')
    p.add_argument('--zip', type=Path, help='--predict: the zip (default: <run>/submission_<folder of images.json>.zip)')
    args = p.parse_args()
    if not (args.predict or args.resume or args.name):
        p.error('training needs --name')
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
    return transforms or None


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


def predict(weights, images, root, size, batch):
    """Each image's boxes ([x1, y1, x2, y2] on the size x size input, as write_submission takes them), classes 0..8
    and scores, at confidence >= 0.001, at most 100 per image; the image's class is its highest-scored box's
    (class 0 if it has none)."""
    from ultralytics import YOLO
    model = YOLO(str(weights))
    preds = {}
    for start in range(0, len(images), batch):
        chunk = images[start:start + batch]
        for im, r in zip(chunk, model.predict([str(root / im['file_name']) for im in chunk], imgsz=size, conf=0.001,
                                              iou=0.7, max_det=100, batch=batch, verbose=False)):
            scale = [size / im['width'], size / im['height']] * 2  # original pixels -> input pixels
            boxes = [[v * f for v, f in zip(box, scale)] for box in r.boxes.xyxy.tolist()]
            classes, scores = r.boxes.cls.int().tolist(), r.boxes.conf.tolist()
            label = classes[max(range(len(scores)), key=scores.__getitem__)] if scores else 0
            preds[im['file_name']] = (label, boxes, classes, scores)
    return preds


def score_holdout(weights, root, size, batch):
    """The holdout's classification and detection scores (train.py's metrics) and its detections."""
    gt_json = root / 'detection' / 'instances_test.json'
    gt = json.loads(gt_json.read_text())
    preds = predict(weights, gt['images'], root, size, batch)
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
                                         args.root or images_json.parent, options['imgsz'], options['batch']),
                         size=options['imgsz'], out=zip_path)
        print(f'{len(images)} images -> {zip_path}')
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
    data_yaml = write_yolo_dataset(args.data)
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
