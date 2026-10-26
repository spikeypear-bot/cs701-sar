"""Train one model jointly for classification and detection, then score it on val and test.
Example: python train.py --backbone vit --init pretrained --adapt lora"""
import argparse
import copy
import csv
import json
import math
import subprocess
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from sarbench.adapters import add_adapters, unfreeze_blocks
from sarbench.backbones import CONVNEXT_MODELS, SWIN_MODELS, build_backbone
from sarbench.data import CHANNELS, SARMultiTask, channel_stats, collate, to_original_xywh
from sarbench.metrics import classification_metrics, detection_metrics
from sarbench.model import DETECTORS, RPN_MATCH, MultiTaskModel, SwinMultiTaskModel
from sarbench.quadfpn import QUAD_FPN_PARTS
from sarbench.simmim import load_ssl
from sarbench.yolo import CLS_HEADS, CLS_HIDDEN, CLS_POOLS, YOLO_MODELS, YOLOMultiTaskModel

# Limits from Codabench (Terms), and the Δm reference: ViT + LoRA on test, mean of 3 seeds (README, section 3)
MAX_PARAMETERS, MAX_TRAINABLE = 130e6, 40e6
REF_F1, REF_MAP = 0.9399, 0.2541
# Cross-entropy of the image class, then the detector's losses: Faster R-CNN's (RPN, then RoI heads), RetinaNet's or
# YOLO's (sarbench/yolo.py)
MAIN_LOSSES = ('classification', 'loss_objectness', 'loss_rpn_box_reg', 'loss_classifier', 'loss_box_reg',
               'loss_focal', 'loss_retina_box_reg', 'loss_yolo_box', 'loss_yolo_cls', 'loss_yolo_dfl', 'loss_yolo_kpt')


def parse_blocks(spec):
    """'0-3,22' -> [0, 1, 2, 3, 22]"""
    blocks = []
    for part in filter(None, spec.split(',')):
        first, _, last = part.partition('-')
        blocks += range(int(first), int(last or first) + 1)
    return blocks


def parse_parts(spec):
    """'all' -> every Quad-FPN part; 'de,bs' -> ['de', 'bs']"""
    return list(QUAD_FPN_PARTS) if spec == 'all' else list(filter(None, spec.split(',')))


def parse_loss_weights(spec):
    """'loss_box_reg=2,classification=0.5' -> {'loss_box_reg': 2.0, 'classification': 0.5}"""
    weights = {}
    for part in filter(None, spec.split(',')):
        name, _, value = part.partition('=')
        weights[name.strip()] = float(value)
    return weights


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, default=Path(__file__).resolve().parents[1] / 'dataset' / 'SARFact-Course-20K')
    p.add_argument('--backbone', required=True,
                   choices=['vit', 'terramind', *SWIN_MODELS, *CONVNEXT_MODELS, *YOLO_MODELS],
                   help='a YOLO model is the whole detector, its own neck and head included (sarbench/yolo.py); it '
                        'takes --adapt full only')
    p.add_argument('--init', required=True, choices=['pretrained', 'scratch'])
    p.add_argument('--adapt', required=True, choices=['full', 'lora', 'moelora'])
    p.add_argument('--lora-rank', type=int, help='rank of each LoRA adapter (MoE-LoRA: of each expert), alpha = 2 x '
                                                 'rank (default: 16, MoE-LoRA 4)')
    p.add_argument('--heatmap-weight', type=float, default=0.0,
                   help='weight of the auxiliary object-centre heatmap loss, training only (default: 0 = no head)')
    p.add_argument('--heatmap-min-sigma', type=float, default=2.0,
                   help='smallest Gaussian sigma of the heatmap targets, in input pixels (default: 2)')
    p.add_argument('--fgfm-levels', type=parse_blocks, default=[],
                   help='Swin only: foreground-guided feature modulation (FGAA-FPN, Ma et al. 2026) on these FPN '
                        'levels, e.g. 0-2 (0: stride 4, 1: 8, 2: 16, 3: 32; about 0.8 M parameters per level; '
                        'default: off); see sarbench/fgfm.py')
    p.add_argument('--fgfm-alpha', type=float, default=0.8,
                   help='modulation strength: lateral map x (1 + alpha x weights) (default: 0.8, as FGAA-FPN)')
    p.add_argument('--fgfm-loss-weight', type=float, default=0.7,
                   help='weight of the foreground mask loss (default: 0.7, as FGAA-FPN)')
    p.add_argument('--detector', choices=DETECTORS, default='faster_rcnn',
                   help='faster_rcnn (RPN + RoI heads) or retinanet (one stage, focal loss), on the same anchors and '
                        'FPN maps (default: faster_rcnn); see sarbench/model.py RetinaNetDetector')
    p.add_argument('--rpn-match', choices=RPN_MATCH, default='iou',
                   help='how the RPN labels anchors positive (>= 0.7) or background (< 0.3) in training: iou, nwd '
                        '(normalized Gaussian Wasserstein distance, Wang et al. 2021: fairer to tiny objects) or max '
                        '(the larger of the two: NWD for tiny boxes, IoU for large ones) (default: iou); see '
                        'sarbench/model.py anchor_similarity')
    p.add_argument('--nwd-c', type=float, default=16.0,
                   help='NWD constant in input pixels (default: 16, the median box size at 512 px; the paper uses the '
                        'mean box size, 35.6 here, but on AI-TOD, where nearly every box is tiny)')
    p.add_argument('--loss-weights', type=parse_loss_weights, default={},
                   help=f'weights of the main losses, e.g. loss_box_reg=2,classification=0.5 (names: '
                        f'{", ".join(MAIN_LOSSES)}; default: all 1)')
    p.add_argument('--input-channels', default='raw',
                   help=f'comma-separated input maps from {CHANNELS}, e.g. raw,nlm,cv (default: raw); see '
                        'sarbench/data.py input_channels')
    p.add_argument('--train-patch-embed', action='store_true',
                   help='with LoRA/MoE-LoRA, also train the input layer (frozen otherwise), so the model can learn '
                        'how to mix --input-channels (Swin-B: 128 x channels x 16 + 384 parameters, its LayerNorm '
                        'included)')
    p.add_argument('--unfreeze-blocks', type=parse_blocks, default=[],
                   help='with LoRA/MoE-LoRA, also fully train these backbone blocks, e.g. 0-3 or 16-21,23 (Swin-B: '
                        '0-1 stage 1, 2-3 stage 2, 4-21 stage 3, 22-23 stage 4; about 0.2, 0.8, 3.2 and 12.6 M '
                        'parameters per block)')
    p.add_argument('--ssl-weights', type=Path,
                   help='ssl.pt of pretrain_ssl.py: its SimMIM-trained LoRA adapters (and blocks) go on the backbone, '
                        'frozen and unmerged, under the --adapt adapters; Swin, LoRA/MoE-LoRA and raw input only')
    p.add_argument('--drop-path', type=float, default=0.0,
                   help='ConvNeXt only: stochastic depth rate, rising linearly over the blocks to this value (the '
                        'ConvNeXt paper, ConvNeXt-B Cascade Mask R-CNN: 0.6; default: 0)')
    p.add_argument('--freeze-layers', type=int, default=0,
                   help='YOLO only: freeze its layers 0..N-1, as Ultralytics\' freeze=N (yolov8l: 6 is the least that '
                        'keeps it under 40 M trainable, 39.7 M; yolov8l-p2: 6 too, 38.9 M; default: 0)')
    p.add_argument('--quad-fpn', type=parse_parts, default=[],
                   help='YOLO only: Quad-FPN (Zhang et al. 2021) in the neck, all or some of de (deformable convs on '
                        'the backbone maps), ca (CARAFE upsampling), pa (space attention before the bottom-up '
                        'down-sampling), bs (balanced pyramid with a non-local block), e.g. all or de,ca; each starts '
                        'as the identity; yolov8m all: 11.35 M parameters (default: off); see sarbench/quadfpn.py')
    p.add_argument('--centre-keypoint', action='store_true',
                   help='YOLO only: train its pose model with one keypoint per box, the box centre (Ultralytics\' pose '
                        'loss, a distance loss, on top of the detection loss; loss_yolo_kpt); predict.py '
                        '--keypoint-centres then moves each box onto its predicted centre; see sarbench/yolo.py')
    p.add_argument('--small-box-assign', type=float, default=0.0,
                   help='YOLO only: YOLO26\'s small-target-aware assignment, box sides below this many input pixels '
                        'widened around their centre for the assigner\'s candidate test only, so boxes between the '
                        'grid points get trained (YOLO26: 16; default: 0 = off); see sarbench/yolo.py')
    p.add_argument('--cls-pool', choices=CLS_POOLS, default='sppf',
                   help='YOLO only: what the image-class head averages: the SPPF map (end of the backbone) or the '
                        'neck maps Detect reads, P3-P5, joined (default: sppf)')
    p.add_argument('--cls-head', choices=CLS_HEADS, default='linear',
                   help=f'YOLO only: LayerNorm -> Linear, or an MLP with one hidden layer of {CLS_HIDDEN} (GELU) '
                        '(default: linear)')
    p.add_argument('--cls-dropout', type=float, default=0.0,
                   help='YOLO only: dropout before the image-class head\'s last Linear (default: 0 = none)')
    p.add_argument('--label-smoothing', type=float, default=0.0,
                   help='label smoothing of the image-class cross-entropy (default: 0)')
    p.add_argument('--unfreeze-lr', type=float, default=2e-5,
                   help='for the pretrained weights of --unfreeze-blocks (default: 2e-5, as for full fine-tuning)')
    p.add_argument('--rot90', action='store_true', help='augment with random 90-degree rotations (training only)')
    p.add_argument('--crop-prob', type=float, default=0.0,
                   help='chance that a training image is a random zoom-in crop: a square of --crop-min-side..1 of '
                        'the side, holding one of its boxes whole, resized back to the input size; boxes cut by it '
                        'are clipped, or dropped if less than half inside (default: 0 = off); see sarbench/data.py '
                        'random_crop')
    p.add_argument('--crop-min-side', type=float, default=0.5,
                   help='smallest crop side, as a share of the image side: zooms up to 1 / this (default: 0.5)')
    p.add_argument('--brightness', type=float, default=0.0,
                   help='brightness and contrast jitter of training images, each uniform in 1 +- this, e.g. 0.2 '
                        '(default: 0 = off); see sarbench/data.py jitter_intensity')
    p.add_argument('--boxpaste-prob', type=float, default=0.0,
                   help='chance that a training image gets same-class BoxPaste objects (default: 0 = off)')
    p.add_argument('--boxpaste-max', type=int, default=3, help='at most this many pasted objects per image')
    p.add_argument('--boxpaste-max-size', type=float, default=32.0,
                   help='only boxes up to this many original pixels per side are copied')
    p.add_argument('--boxpaste-classes', default='ship,wind_turbine,oil_tank,car,aircraft',
                   help='comma-separated classes BoxPaste copies (default: the compact ones; bridge, playground, '
                        'harbor and airport need their surroundings)')
    p.add_argument('--image-size', type=int, default=512,
                   help='side of the square model input in pixels, a multiple of 32; images are stretched to it and '
                        'the anchors scale with it (default: 512; ViT, Swin and YOLO)')
    p.add_argument('--epochs', type=int, default=24)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--accum-steps', type=int, default=1,
                   help='optimizer step every N batches, gradients summed over them: --batch-size 8 --accum-steps 2 '
                        'trains as batch 16 in half the memory (default: 1)')
    p.add_argument('--schedule', choices=['cosine', 'step'], default='cosine',
                   help='after one epoch of linear warmup: cosine decay to 0, or the ConvNeXt paper\'s step schedule, '
                        'lr x 0.1 at 3/4 and 11/12 of the epochs (27 and 33 of 36) (default: cosine)')
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--backbone-lr', type=float, help='for the trainable backbone parameters (default: --lr)')
    p.add_argument('--weight-decay', type=float, default=0.05)
    p.add_argument('--optimizer', choices=['adamw', 'sgd'], default='adamw',
                   help='adamw (weight decay on every parameter, unless --no-decay-1d), or SGD with Nesterov momentum '
                        'and, as Ultralytics, no weight decay on 1-D parameters (norm weights, biases); YOLO\'s SGD '
                        'lr: 0.01 (default: adamw)')
    p.add_argument('--momentum', type=float, default=0.937, help='SGD momentum (default: 0.937, Ultralytics\')')
    p.add_argument('--no-decay-1d', action='store_true',
                   help='AdamW: no weight decay on 1-D parameters (norm weights, biases), as SGD always does')
    p.add_argument('--ema', type=float, default=0.0,
                   help='weight EMA decay, e.g. 0.9999 (Ultralytics\' ModelEMA: the decay ramps up over the first '
                        'few thousand optimizer steps); validation, test and model.pt then use the EMA weights '
                        '(default: 0 = off)')
    p.add_argument('--eval-every', type=int, default=4, help='validate every N epochs, and after the last one')
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--limit', type=int, help='use only the first N images of every split (quick checks)')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out', type=Path, default=Path('runs'))
    p.add_argument('--name', help='run name; the run folder is <out>/<name>_<time> (default: backbone_init_adapt)')
    args = p.parse_args()
    if args.init == 'scratch' and args.adapt != 'full':
        p.error('--init scratch is only valid with --adapt full')
    if args.backbone_lr is None:
        args.backbone_lr = args.lr
    if args.unfreeze_blocks and args.adapt == 'full':
        p.error('--unfreeze-blocks is for LoRA and MoE-LoRA; --adapt full trains every block already')
    if args.train_patch_embed and args.backbone == 'terramind':
        p.error('--train-patch-embed is for the ViT and Swin backbones')
    if args.fgfm_levels and args.backbone not in SWIN_MODELS:
        p.error('--fgfm-levels is for the Swin backbones (an FPN over 4 backbone maps)')
    if not set(args.fgfm_levels) <= {0, 1, 2, 3}:
        p.error('--fgfm-levels takes levels 0 to 3')
    if not 0 <= args.crop_prob <= 1 or not 0 < args.crop_min_side <= 1:
        p.error('--crop-prob takes 0 to 1, --crop-min-side more than 0 to 1')
    if args.quad_fpn and args.backbone not in YOLO_MODELS:
        p.error('--quad-fpn is for the YOLO backbones')
    if args.centre_keypoint and args.backbone not in YOLO_MODELS:
        p.error('--centre-keypoint is for the YOLO backbones')
    if args.backbone.endswith('-p2') and (args.quad_fpn or args.centre_keypoint):
        p.error('the -p2 YOLO models take neither --quad-fpn nor --centre-keypoint')
    if not set(args.quad_fpn) <= set(QUAD_FPN_PARTS):
        p.error(f'--quad-fpn takes all or parts from {QUAD_FPN_PARTS}')
    if not set(args.input_channels.split(',')) <= set(CHANNELS):
        p.error(f'--input-channels takes names from {CHANNELS}')
    if args.detector == 'retinanet' and args.rpn_match != 'iou':
        p.error('--rpn-match is for Faster R-CNN: RetinaNet has no RPN')
    if args.ssl_weights and (args.backbone not in SWIN_MODELS or args.adapt == 'full'
                             or args.init != 'pretrained' or args.input_channels != 'raw'):
        p.error('--ssl-weights needs a pretrained Swin backbone, --adapt lora or moelora and --input-channels raw')
    if args.image_size % 32 or (args.image_size != 512 and args.backbone == 'terramind'):
        p.error('--image-size must be a multiple of 32 (TerraMind: 512 only)')
    if args.backbone in YOLO_MODELS and (args.adapt != 'full' or args.detector != 'faster_rcnn'
                                         or args.rpn_match != 'iou' or args.heatmap_weight or args.train_patch_embed):
        p.error('a YOLO backbone takes --adapt full and none of --detector, --rpn-match, --heatmap-weight, '
                '--train-patch-embed: it is a whole detector of its own')
    if args.drop_path and args.backbone not in CONVNEXT_MODELS:
        p.error('--drop-path is for the ConvNeXt backbones')
    if args.freeze_layers and args.backbone not in YOLO_MODELS:
        p.error('--freeze-layers is for the YOLO backbones')
    if not set(args.loss_weights) <= set(MAIN_LOSSES):
        p.error(f'--loss-weights takes names from {MAIN_LOSSES} (the auxiliary losses have their own flags)')
    yolo_head = args.small_box_assign or args.cls_pool != 'sppf' or args.cls_head != 'linear' or args.cls_dropout
    if yolo_head and args.backbone not in YOLO_MODELS:
        p.error('--small-box-assign, --cls-pool, --cls-head and --cls-dropout are for the YOLO backbones')
    if not all(0 <= v < 1 for v in (args.ema, args.cls_dropout, args.label_smoothing, args.brightness)):
        p.error('--ema, --cls-dropout, --label-smoothing and --brightness take 0 to less than 1')
    return args


def log(message, file):
    """Print to the console and to the run's log.txt."""
    print(message, flush=True)
    print(message, file=file, flush=True)


def make_loader(args, split):
    """Batches of `split`; only the training set is shuffled and augmented."""
    train = split == 'train'
    boxpaste = {'prob': args.boxpaste_prob, 'max_pastes': args.boxpaste_max, 'max_size': args.boxpaste_max_size,
                'classes': args.boxpaste_classes.split(',')} if args.boxpaste_prob > 0 else None
    dataset = SARMultiTask(args.data, split, train=train, size=args.image_size, limit=args.limit,
                           channels=args.input_channels.split(','), stats=args.channel_stats, boxpaste=boxpaste,
                           rot90=args.rot90, crop_prob=args.crop_prob, crop_min_side=args.crop_min_side,
                           brightness=args.brightness)
    return DataLoader(dataset, args.batch_size, shuffle=train, num_workers=args.workers, collate_fn=collate,
                      persistent_workers=train and args.workers > 0)  # training workers live across epochs


def lr_factor(step, warmup, total, milestones=None):
    """Learning-rate multiplier: linear warmup over `warmup` steps, then cosine decay to 0 at step `total`, or, with
    milestones (steps), x 0.1 at each milestone passed."""
    if step < warmup:
        return (step + 1) / warmup
    if milestones is not None:
        return 0.1 ** sum(step >= m for m in milestones)
    return 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(total - warmup, 1)))  # --epochs 1: no decay phase


def make_optimizer(groups, args):
    """AdamW over the parameter groups, weight decay on every parameter; or SGD with Nesterov momentum. With SGD, or
    AdamW and args.no_decay_1d, each group's 1-D parameters (norm weights, biases) move to a group of their own at
    the same lr without weight decay, as Ultralytics does."""
    no_decay = []
    if args.optimizer == 'sgd' or args.no_decay_1d:
        for group in groups:
            flat = [p for p in group['params'] if p.ndim <= 1]
            group['params'] = [p for p in group['params'] if p.ndim > 1]
            if flat:
                no_decay.append({**group, 'params': flat, 'weight_decay': 0.0})
    # appended after, so group 1 stays the new modules' (the lr in the log)
    if args.optimizer == 'adamw':
        return torch.optim.AdamW(groups + no_decay, lr=args.lr, weight_decay=args.weight_decay)
    return torch.optim.SGD(groups + no_decay, lr=args.lr, momentum=args.momentum, nesterov=True,
                           weight_decay=args.weight_decay)


class EMA:
    """Ultralytics' ModelEMA: a copy of the model whose floating-point weights and buffers (BatchNorm statistics
    included) follow the trained ones, ema = d x ema + (1 - d) x model after every optimizer step, with
    d = decay x (1 - exp(-updates / tau)): small at first, so the copy does not stay near the starting weights."""

    def __init__(self, model, decay, tau=2000):
        self.model = copy.deepcopy(model).eval().requires_grad_(False)
        self.decay, self.tau, self.updates = decay, tau, 0

    @torch.no_grad()
    def update(self, model):
        self.updates += 1
        d = self.decay * (1 - math.exp(-self.updates / self.tau))
        for ema, trained in zip(self.model.state_dict().values(), model.state_dict().values()):
            if ema.dtype.is_floating_point:  # not BatchNorm's num_batches_tracked
                ema.mul_(d).add_(trained, alpha=1 - d)


def train_one_epoch(model, loader, optimizer, schedule, epoch, log_file, weights, accum_steps=1, label_smoothing=0.0,
                    ema=None):
    """One pass over the training set; the loss is cross-entropy (label_smoothing) plus the detector's losses, plus
    the auxiliary losses the model returns ('heatmap', 'foreground'), each times weights[name]. The log shows each
    loss before weighting. One optimizer step per accum_steps batches, on the mean of their losses (the last step of
    an epoch may have fewer), each followed by an EMA update if ema is given."""
    model.train()
    start, seen = time.time(), 0
    for step, (images, labels, targets) in enumerate(loader, 1):
        targets = [{'boxes': t['boxes'].cuda(), 'labels': t['labels'].cuda()} for t in targets]
        with torch.autocast('cuda', dtype=torch.bfloat16):  # bf16 has fp32's range: no loss scaling needed
            logits, _, det_losses = model(images.cuda(), targets)
            losses = {'classification': F.cross_entropy(logits, labels.cuda(), label_smoothing=label_smoothing),
                      **det_losses}
        loss = sum(weights.get(name, 1.0) * value for name, value in losses.items())
        (loss / accum_steps).backward()
        if step % accum_steps == 0 or step == len(loader):
            optimizer.step()
            optimizer.zero_grad()
            schedule.step()
            if ema is not None:
                ema.update(model)
        seen += len(images)
        if step % 50 == 0 or step == len(loader):
            parts = ' '.join(f'{name} {value.item():.3f}' for name, value in losses.items())
            log(f'epoch {epoch} iter {step}/{len(loader)} loss {loss.item():.3f} ({parts}) '
                f'lr {optimizer.param_groups[1]["lr"]:.2e} {seen / (time.time() - start):.1f} images/s', log_file)
            start, seen = time.time(), 0


@torch.no_grad()
def evaluate(model, loader, gt_json, size):
    """Classification and COCO detection metrics on one split, and the detections in COCO result format; size: the
    side of the model input."""
    model.eval()
    labels, predicted, detections, image_ids = [], [], [], []
    for images, y, targets in loader:
        with torch.autocast('cuda', dtype=torch.bfloat16):
            logits, outputs, _ = model(images.cuda())
        labels += y.tolist()
        predicted += logits.argmax(dim=1).tolist()
        for target, out in zip(targets, outputs):
            image_ids.append(target['image_id'])
            boxes = to_original_xywh(out['boxes'], target['orig_size'], size).tolist()
            detections += [{'image_id': target['image_id'], 'category_id': c, 'bbox': b, 'score': s}
                           for b, c, s in zip(boxes, out['labels'].tolist(), out['scores'].tolist())]
    metrics = {**classification_metrics(labels, predicted), **detection_metrics(gt_json, detections, image_ids)}
    return metrics, detections


def headline(metrics):
    return ' '.join(f'{k} {metrics[k]:.4f}' for k in ('accuracy', 'macro_f1', 'balanced_accuracy', 'mAP', 'AP50'))


def save_code_version(run_dir):
    """The git commit, and uncommitted changes to tracked files as code.diff, so every run can be reproduced."""
    git = lambda *cmd: subprocess.run(['git', *cmd], capture_output=True, text=True).stdout
    diff = git('diff', 'HEAD')
    if diff:
        (run_dir / 'code.diff').write_text(diff)
    return git('rev-parse', 'HEAD').strip() or None, bool(diff)


def append_results(summary, path):
    """One row per run in a table shared by all runs. Δm is rough here: the reference was scored on the
    official test split, our numbers on the local holdout."""
    final = summary['test']
    row = {'run': summary['run'], 'commit': summary['commit'], 'dirty': summary['dirty'],
           'macro_f1': round(final['macro_f1'], 4), 'mAP': round(final['mAP'], 4),
           'dm_rough': round(100 * ((final['macro_f1'] - REF_F1) / REF_F1 + (final['mAP'] - REF_MAP) / REF_MAP) / 2, 2),
           'parameters_M': round(summary['parameters'] / 1e6, 2),
           'trainable_M': round(summary['trainable_parameters'] / 1e6, 2),
           'train_hours': round(summary['train_hours'], 2)}
    new = not path.exists()
    with open(path, 'a', newline='') as f:
        writer = csv.DictWriter(f, list(row))
        if new:
            writer.writeheader()
        writer.writerow(row)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    name = args.name or f'{args.backbone}_{args.init}_{args.adapt}'
    run_dir = args.out / f'{name}_{time.strftime("%Y%m%d-%H%M%S")}'  # a new folder per run: nothing is overwritten
    run_dir.mkdir(parents=True)
    log_file = open(run_dir / 'log.txt', 'w')
    log(json.dumps(vars(args), default=str), log_file)
    commit, dirty = save_code_version(run_dir)
    log(f'code: commit {commit}' + (', with uncommitted changes (code.diff)' if dirty else ''), log_file)

    # Data. The input channels' mean and std are saved with the run (metrics.json args), for predict.py
    args.channel_stats = channel_stats(args.data, args.input_channels.split(','))
    log(f'input channels {args.input_channels}: mean, std {args.channel_stats}', log_file)
    train_loader, val_loader, test_loader = (make_loader(args, split) for split in ('train', 'val', 'test'))

    # Model: adapters are created on the CPU, so they are added before .cuda()
    unfrozen, ssl_trained = [], []
    if args.backbone in YOLO_MODELS:  # a whole detector, every parameter trained (sarbench/yolo.py)
        model = YOLOMultiTaskModel(args.backbone, pretrained=args.init == 'pretrained',
                                   in_chans=len(args.input_channels.split(',')),
                                   freeze_layers=args.freeze_layers, quad_fpn=args.quad_fpn,
                                   centre_keypoint=args.centre_keypoint, small_box_assign=args.small_box_assign,
                                   cls_pool=args.cls_pool, cls_head=args.cls_head,
                                   cls_dropout=args.cls_dropout).cuda()
        log(f'{args.backbone}: {model.transferred} tensors from the COCO weights', log_file)
    else:
        backbone = build_backbone(args.backbone, pretrained=args.init == 'pretrained',
                                  in_chans=len(args.input_channels.split(',')), img_size=args.image_size,
                                  drop_path=args.drop_path)
        if args.ssl_weights:  # before the task adapters, which wrap the SSL ones
            checkpoint = torch.load(args.ssl_weights, map_location='cpu')
            if checkpoint['backbone'] != args.backbone:
                raise ValueError(f'{args.ssl_weights} is for {checkpoint["backbone"]}, not {args.backbone}')
            ssl_trained = load_ssl(backbone, checkpoint)
            args.ssl_rank = checkpoint['rank']  # in metrics.json, for predict.py
            log(f'SSL weights {args.ssl_weights}: LoRA rank {checkpoint["rank"]}, blocks '
                f'{checkpoint["unfreeze_blocks"]} unfrozen, {checkpoint["trained_parameters"]:,} parameters trained',
                log_file)
        if args.adapt != 'full':
            add_adapters(backbone, args.adapt, args.lora_rank)  # also freezes the backbone
            if args.train_patch_embed:
                backbone.patch_embed.requires_grad_(True)  # trained at --backbone-lr with the adapters
            unfrozen = unfreeze_blocks(backbone, args.unfreeze_blocks)  # trained at --unfreeze-lr
        multiscale = args.backbone in SWIN_MODELS or args.backbone in CONVNEXT_MODELS  # 4 maps -> FPN
        model_class = SwinMultiTaskModel if multiscale else MultiTaskModel
        fgfm = dict(fgfm_levels=args.fgfm_levels, fgfm_alpha=args.fgfm_alpha) if args.fgfm_levels else {}  # Swin only
        model = model_class(backbone, task_routing=args.adapt == 'moelora', heatmap=args.heatmap_weight > 0,
                            heatmap_min_sigma=args.heatmap_min_sigma, rpn_match=args.rpn_match, nwd_c=args.nwd_c,
                            detector=args.detector, image_size=args.image_size, **fgfm).cuda()
    parameters = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f'parameters: {parameters:,} total, {trainable:,} trainable', log_file)
    trained_overall = trainable  # with SSL: plus what it trained that stays frozen here, its decoder included
    if args.ssl_weights:
        trained_overall += (sum(p.numel() for p in ssl_trained if not p.requires_grad)
                            + checkpoint['pretraining_only_parameters'])
        log(f'trained overall, with SSL: {trained_overall:,}', log_file)
    if parameters > MAX_PARAMETERS or trained_overall > MAX_TRAINABLE:
        log(f'WARNING: over the limits ({MAX_PARAMETERS / 1e6:.0f} M total, {MAX_TRAINABLE / 1e6:.0f} M trainable); '
            'not allowed for a submission', log_file)

    # Optimizer and schedule: trainable backbone parameters at --backbone-lr (unfrozen blocks' pretrained weights at
    # --unfreeze-lr), all new modules at --lr. .cuda() keeps the parameter objects, so `unfrozen` still matches
    unfrozen_ids = {id(p) for p in unfrozen}
    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad and id(p) not in unfrozen_ids]
    head_params = [p for name, p in model.named_parameters() if not name.startswith('backbone.')]
    groups = [{'params': backbone_params, 'lr': args.backbone_lr}, {'params': head_params}]
    if unfrozen:
        groups.append({'params': unfrozen, 'lr': args.unfreeze_lr})
    optimizer = make_optimizer(groups, args)
    steps_per_epoch = math.ceil(len(train_loader) / args.accum_steps)  # optimizer steps
    warmup, total = steps_per_epoch, args.epochs * steps_per_epoch  # one epoch of warmup
    # step: the paper's epochs 27 and 33 of 36 (mmdet lr_config step=[27, 33]), as shares of --epochs
    milestones = [args.epochs * 3 // 4 * steps_per_epoch,
                  args.epochs * 11 // 12 * steps_per_epoch] if args.schedule == 'step' else None
    schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: lr_factor(step, warmup, total, milestones))

    # With --ema, the EMA copy is what is validated, tested and saved
    ema = EMA(model, args.ema) if args.ema else None
    final_model = ema.model if ema else model
    if ema:
        log(f'EMA: decay {args.ema}, tau {ema.tau} steps; validation, test and model.pt use the EMA weights', log_file)

    # Train, validating every --eval-every epochs; the reported model is the last-epoch model
    history, start = [], time.time()
    for epoch in range(1, args.epochs + 1):
        train_one_epoch(model, train_loader, optimizer, schedule, epoch, log_file,
                        {'heatmap': args.heatmap_weight, 'foreground': args.fgfm_loss_weight, **args.loss_weights},
                        args.accum_steps, args.label_smoothing, ema)
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            val, val_detections = evaluate(final_model, val_loader, args.data / 'detection' / 'instances_val.json',
                                           args.image_size)
            history.append({'epoch': epoch, **val})
            log(f'epoch {epoch} val: {headline(val)}', log_file)
    train_hours = (time.time() - start) / 3600

    # Evaluate on test, once
    test, test_detections = evaluate(final_model, test_loader, args.data / 'detection' / 'instances_test.json',
                                     args.image_size)
    peak_gb = torch.cuda.max_memory_allocated() / 2**30
    log(f'test: {headline(test)}\ntraining {train_hours:.2f} h, peak GPU memory {peak_gb:.1f} GB', log_file)

    # Save
    summary = {'run': run_dir.name, 'commit': commit, 'dirty': dirty,
               'args': vars(args), 'parameters': parameters, 'trainable_parameters': trainable,
               'trained_overall_parameters': trained_overall,
               'train_hours': train_hours, 'peak_gpu_memory_gb': peak_gb,
               'val_history': history, 'val': history[-1], 'test': test}
    (run_dir / 'metrics.json').write_text(json.dumps(summary, indent=2, default=str))
    (run_dir / 'predictions_val.json').write_text(json.dumps(val_detections))
    (run_dir / 'predictions_test.json').write_text(json.dumps(test_detections))
    torch.save(final_model.state_dict(), run_dir / 'model.pt')
    append_results(summary, args.out / 'results.csv')


if __name__ == '__main__':
    main()
