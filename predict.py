"""Predict images with a trained run and write the submission zip for Codabench.
Example: python predict.py runs/<run> cs701-sar-course-data/val/images.json
On a file with labels, such as the local holdout (data/detection/instances_val.json --root data), it also scores
the zip: a check of the whole path to the zip, which should give the run's own test scores.
--class-filter uses that every image holds boxes of one class only: hard keeps only the boxes of the image's predicted
class; soft multiplies each box's score by the image classifier's probability of the box's class, so a wrong-class box
sinks when the classifier is sure, and both classes keep their boxes when it is not."""
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
from sarbench.backbones import CONVNEXT_MODELS, SWIN_MODELS, build_backbone
from sarbench.data import SARImages, collate_images
from sarbench.metrics import classification_metrics, detection_metrics
from sarbench.model import MultiTaskModel, SwinMultiTaskModel
from sarbench.yolo import NMS, YOLO_MODELS, YOLOMultiTaskModel
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
    p.add_argument('--nms', choices=NMS, help='YOLO runs only: hard (NMS, as in training) or soft (Gaussian Soft-NMS: '
                                              'overlapping boxes of a class are ranked lower, not dropped)')
    p.add_argument('--nms-thresh', type=float, help='YOLO, hard NMS: IoU above which a box is dropped (default: 0.7)')
    p.add_argument('--soft-sigma', type=float, help='YOLO, soft NMS: scores times exp(-IoU^2 / sigma) (default: 0.5, '
                                                    'as Bodla et al.)')
    p.add_argument('--score-thresh', type=float, help='YOLO: lowest score kept (default: 0.001)')
    p.add_argument('--keypoint-centres', action='store_true',
                   help='YOLO runs with --centre-keypoint only: move each box onto its predicted centre keypoint')
    p.add_argument('--class-filter', choices=('hard', 'soft'),
                   help='any run: keep only the boxes of the predicted image class (hard), or multiply each box score '
                        'by the image-class probability of its class (soft) (default: off)')
    args = p.parse_args()
    args.root = args.root or args.images.parent
    # the post-processing options given, set on the model; they go in the default zip name, so that the zip of the
    # run's own settings is not overwritten
    args.postprocess = {k: v for k, v in (('nms', args.nms), ('nms_thresh', args.nms_thresh),
                                          ('soft_sigma', args.soft_sigma), ('score_thresh', args.score_thresh),
                                          ('keypoint_centres', args.keypoint_centres or None))
                        if v is not None}
    suffix = ''.join(f'_{k}-{v}' for k, v in (*args.postprocess.items(), ('class_filter', args.class_filter))
                     if v is not None)
    args.out = args.out or args.run / f'submission_{args.images.parent.name}{suffix}.zip'
    return args


def load_model(run):
    """The model of a train.py run, rebuilt from the options in its metrics.json, with the weights of model.pt."""
    options = json.loads((run / 'metrics.json').read_text())['args']
    size = options.get('image_size', 512)  # runs from before --image-size: 512
    state = torch.load(run / 'model.pt', map_location='cpu')
    if options['backbone'] in YOLO_MODELS:  # the whole detector, rebuilt from its Ultralytics config
        model = YOLOMultiTaskModel(options['backbone'], pretrained=False,
                                   in_chans=len(options.get('input_channels', 'raw').split(',')),
                                   quad_fpn=options.get('quad_fpn', []),  # runs before --quad-fpn: none
                                   centre_keypoint=options.get('centre_keypoint', False),
                                   # runs before --cls-pool/--cls-head/--cls-dropout: the plain head
                                   cls_pool=options.get('cls_pool', 'sppf'), cls_head=options.get('cls_head', 'linear'),
                                   cls_dropout=options.get('cls_dropout', 0.0))
        model.load_state_dict(state)
        return model.cuda().eval()
    backbone = build_backbone(options['backbone'], pretrained=False,  # every weight comes from model.pt
                              in_chans=len(options.get('input_channels', 'raw').split(',')), img_size=size)
    if options.get('ssl_weights'):  # the SSL adapters under the task ones; their weights are in model.pt too
        add_adapters(backbone, 'lora', options['ssl_rank'])
    if options['adapt'] != 'full':
        add_adapters(backbone, options['adapt'], options.get('lora_rank'))  # runs before --lora-rank: the default
    multiscale = options['backbone'] in SWIN_MODELS or options['backbone'] in CONVNEXT_MODELS  # 4 maps -> FPN
    model_class = SwinMultiTaskModel if multiscale else MultiTaskModel
    # FGFM runs at inference, so it is rebuilt; runs from before --fgfm-levels have no such option
    fgfm = {k: options[k] for k in ('fgfm_levels', 'fgfm_alpha')} if options.get('fgfm_levels') else {}
    # Runs from before --detector are Faster R-CNN; no heatmap head: training only
    model = model_class(backbone, task_routing=options['adapt'] == 'moelora',
                        detector=options.get('detector', 'faster_rcnn'), image_size=size, **fgfm)
    model.load_state_dict({k: v for k, v in state.items() if not k.startswith('heatmap_head.')})
    return model.cuda().eval()


@torch.no_grad()
def predict(model, loader, class_filter=None):
    """preds[file_name] = (label 0..8, boxes XYXY on the model input, classes 0..8, scores), as write_submission
    takes them. class_filter: None, 'hard' or 'soft', see the module docstring."""
    preds = {}
    for images, infos in loader:
        with torch.autocast('cuda', dtype=torch.bfloat16):
            logits, outputs, _ = model(images.cuda())
        for info, probabilities, out in zip(infos, logits.float().softmax(dim=1), outputs):
            label = int(probabilities.argmax())
            boxes, classes, scores = out['boxes'], out['labels'] - 1, out['scores']  # detector labels 1..9 (0: none)
            if class_filter == 'hard':
                keep = classes == label
                boxes, classes, scores = boxes[keep], classes[keep], scores[keep]
            elif class_filter == 'soft':
                scores = scores * probabilities[classes]
            preds[info['file_name']] = (label, boxes.tolist(), classes.tolist(), scores.tolist())
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
    size = options.get('image_size', 512)
    dataset = SARImages(args.images, args.root, size=size, channels=options.get('input_channels', 'raw').split(','),
                        stats=options.get('channel_stats'))
    loader = DataLoader(dataset, args.batch_size, num_workers=args.workers, collate_fn=collate_images)
    model = load_model(args.run)
    if args.postprocess and not isinstance(model, YOLOMultiTaskModel):
        raise SystemExit('--nms, --nms-thresh, --soft-sigma, --score-thresh and --keypoint-centres are for YOLO runs')
    if args.keypoint_centres and not model.centre_keypoint:
        raise SystemExit('--keypoint-centres needs a run trained with --centre-keypoint')
    for name, value in args.postprocess.items():
        setattr(model, name, value)
    preds = predict(model, loader, args.class_filter)
    write_submission(dataset.images, preds, size=size, out=args.out)
    print(f'{len(preds)} images -> {args.out}')
    if 'annotations' in json.loads(args.images.read_text()):
        print(f'scores: {headline(score(args.out, args.images))}')


if __name__ == '__main__':
    main()
