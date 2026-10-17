"""Predict images with a trained run and write the submission zip for Codabench.
Example: python predict.py runs/<run> cs701-sar-course-data/val/images.json
On a file with labels, such as the local holdout (data/detection/instances_val.json --root data), it also scores
the zip: a check of the whole path to the zip, which should give the run's own test scores."""
import argparse
import csv
import io
import json
import zipfile
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from make_submission import write_submission
from sarbench.adapters import add_adapters
from sarbench.backbones import SWIN_MODELS, build_backbone
from sarbench.data import SARImages, collate_images
from sarbench.metrics import classification_metrics, detection_metrics
from sarbench.model import MultiTaskModel, SwinMultiTaskModel
from train import headline


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('run', type=Path, help='a run folder of train.py, with model.pt and metrics.json')
    p.add_argument('images', type=Path, help='the images to predict, e.g. cs701-sar-course-data/val/images.json')
    p.add_argument('--root', type=Path, help='the folder the file names in images.json are relative to '
                                             '(default: the folder of images.json)')
    p.add_argument('--out', type=Path, help='default: <run>/submission_<folder of images.json>.zip')
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--workers', type=int, default=8)
    args = p.parse_args()
    args.root = args.root or args.images.parent
    args.out = args.out or args.run / f'submission_{args.images.parent.name}.zip'
    return args


def load_model(run):
    """The model of a train.py run, rebuilt from the options in its metrics.json, with the weights of model.pt."""
    options = json.loads((run / 'metrics.json').read_text())['args']
    backbone = build_backbone(options['backbone'], pretrained=False,  # every weight comes from model.pt
                              in_chans=len(options.get('input_channels', 'raw').split(',')))
    if options['adapt'] != 'full':
        add_adapters(backbone, options['adapt'], options.get('lora_rank'))  # runs before --lora-rank: the default
    model_class = SwinMultiTaskModel if options['backbone'] in SWIN_MODELS else MultiTaskModel
    model = model_class(backbone, task_routing=options['adapt'] == 'moelora')  # no heatmap head: training only
    state = torch.load(run / 'model.pt', map_location='cpu')
    model.load_state_dict({k: v for k, v in state.items() if not k.startswith('heatmap_head.')})
    return model.cuda().eval()


@torch.no_grad()
def predict(model, loader):
    """preds[file_name] = (label 0..8, boxes XYXY on the 512 px input, classes 0..8, scores), as write_submission
    takes them."""
    preds = {}
    for images, infos in loader:
        with torch.autocast('cuda', dtype=torch.bfloat16):
            logits, outputs, _ = model(images.cuda())
        for info, label, out in zip(infos, logits.argmax(dim=1).tolist(), outputs):
            classes = (out['labels'] - 1).tolist()  # the detector's labels are 1..9 (0 = background)
            preds[info['file_name']] = (label, out['boxes'].tolist(), classes, out['scores'].tolist())
    return preds


def score(zip_path, gt_json):
    """The scores of a submission zip against a COCO file with annotations."""
    with zipfile.ZipFile(zip_path) as z:
        rows = csv.DictReader(io.TextIOWrapper(z.open('classification.csv')))
        predicted = {int(r['image_id']): int(r['label']) for r in rows}
        detections = json.loads(z.read('detection.json'))
    gt = json.loads(gt_json.read_text())
    truth = {a['image_id']: a['category_id'] - 1 for a in gt['annotations']}  # all boxes of an image: its class
    ids = [img['id'] for img in gt['images']]
    return {**classification_metrics([truth[i] for i in ids], [predicted[i] for i in ids]),
            **detection_metrics(gt_json, detections, ids)}


def main():
    args = parse_args()
    options = json.loads((args.run / 'metrics.json').read_text())['args']  # the run's input channels and their stats
    dataset = SARImages(args.images, args.root, channels=options.get('input_channels', 'raw').split(','),
                        stats=options.get('channel_stats'))
    loader = DataLoader(dataset, args.batch_size, num_workers=args.workers, collate_fn=collate_images)
    preds = predict(load_model(args.run), loader)
    write_submission(dataset.images, preds, out=args.out)
    print(f'{len(preds)} images -> {args.out}')
    if 'annotations' in json.loads(args.images.read_text()):
        print(f'scores: {headline(score(args.out, args.images))}')


if __name__ == '__main__':
    main()
