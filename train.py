"""Train one model jointly for classification and detection, then score it on val and test.
Example: python train.py --backbone vit --init pretrained --adapt lora"""
import argparse
import csv
import json
import math
import subprocess
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from sarbench.adapters import add_adapters
from sarbench.backbones import SWIN_MODELS, build_backbone
from sarbench.data import CHANNELS, SARMultiTask, channel_stats, collate, to_original_xywh
from sarbench.metrics import classification_metrics, detection_metrics
from sarbench.model import MultiTaskModel, SwinMultiTaskModel

# Limits from Codabench (Terms), and the Δm reference: ViT + LoRA on test, mean of 3 seeds (README, section 3)
MAX_PARAMETERS, MAX_TRAINABLE = 130e6, 40e6
REF_F1, REF_MAP = 0.9399, 0.2541


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, default=Path(__file__).resolve().parents[1] / 'dataset' / 'SARFact-Course-20K')
    p.add_argument('--backbone', required=True, choices=['vit', 'terramind', *SWIN_MODELS])
    p.add_argument('--init', required=True, choices=['pretrained', 'scratch'])
    p.add_argument('--adapt', required=True, choices=['full', 'lora', 'moelora'])
    p.add_argument('--lora-rank', type=int, help='rank of each LoRA adapter (MoE-LoRA: of each expert), alpha = 2 x '
                                                 'rank (default: 16, MoE-LoRA 4)')
    p.add_argument('--heatmap-weight', type=float, default=0.0,
                   help='weight of the auxiliary object-centre heatmap loss, training only (default: 0 = no head)')
    p.add_argument('--heatmap-min-sigma', type=float, default=2.0,
                   help='smallest Gaussian sigma of the heatmap targets, in input pixels (default: 2)')
    p.add_argument('--input-channels', default='raw',
                   help=f'comma-separated input maps from {CHANNELS}, e.g. raw,nlm,cv (default: raw); see '
                        'sarbench/data.py input_channels')
    p.add_argument('--train-patch-embed', action='store_true',
                   help='with LoRA/MoE-LoRA, also train the input layer (frozen otherwise), so the model can learn '
                        'how to mix --input-channels (Swin-B: 128 x channels x 16 + 384 parameters, its LayerNorm '
                        'included)')
    p.add_argument('--rot90', action='store_true', help='augment with random 90-degree rotations (training only)')
    p.add_argument('--boxpaste-prob', type=float, default=0.0,
                   help='chance that a training image gets same-class BoxPaste objects (default: 0 = off)')
    p.add_argument('--boxpaste-max', type=int, default=3, help='at most this many pasted objects per image')
    p.add_argument('--boxpaste-max-size', type=float, default=32.0,
                   help='only boxes up to this many original pixels per side are copied')
    p.add_argument('--boxpaste-classes', default='ship,wind_turbine,oil_tank,car,aircraft',
                   help='comma-separated classes BoxPaste copies (default: the compact ones; bridge, playground, '
                        'harbor and airport need their surroundings)')
    p.add_argument('--epochs', type=int, default=24)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--backbone-lr', type=float, help='for the trainable backbone parameters (default: --lr)')
    p.add_argument('--weight-decay', type=float, default=0.05)
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
    if args.train_patch_embed and args.backbone == 'terramind':
        p.error('--train-patch-embed is for the ViT and Swin backbones')
    if not set(args.input_channels.split(',')) <= set(CHANNELS):
        p.error(f'--input-channels takes names from {CHANNELS}')
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
    dataset = SARMultiTask(args.data, split, train=train, limit=args.limit, channels=args.input_channels.split(','),
                           stats=args.channel_stats, boxpaste=boxpaste, rot90=args.rot90)
    return DataLoader(dataset, args.batch_size, shuffle=train, num_workers=args.workers, collate_fn=collate,
                      persistent_workers=train and args.workers > 0)  # training workers live across epochs


def lr_factor(step, warmup, total):
    """Learning-rate multiplier: linear warmup over `warmup` steps, then cosine decay to 0 at step `total`."""
    if step < warmup:
        return (step + 1) / warmup
    return 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(total - warmup, 1)))  # --epochs 1: no decay phase


def train_one_epoch(model, loader, optimizer, schedule, epoch, log_file, heatmap_weight=0.0):
    """One pass over the training set; the loss is cross-entropy plus the four Faster R-CNN losses, plus
    heatmap_weight x the heatmap loss with the heatmap head. The log shows each loss before weighting."""
    model.train()
    start, seen = time.time(), 0
    for step, (images, labels, targets) in enumerate(loader, 1):
        targets = [{'boxes': t['boxes'].cuda(), 'labels': t['labels'].cuda()} for t in targets]
        with torch.autocast('cuda', dtype=torch.bfloat16):  # bf16 has fp32's range: no loss scaling needed
            logits, _, det_losses = model(images.cuda(), targets)
            losses = {'classification': F.cross_entropy(logits, labels.cuda()), **det_losses}
        loss = sum(heatmap_weight * value if name == 'heatmap' else value for name, value in losses.items())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        schedule.step()
        seen += len(images)
        if step % 50 == 0 or step == len(loader):
            parts = ' '.join(f'{name} {value.item():.3f}' for name, value in losses.items())
            log(f'epoch {epoch} iter {step}/{len(loader)} loss {loss.item():.3f} ({parts}) '
                f'lr {optimizer.param_groups[1]["lr"]:.2e} {seen / (time.time() - start):.1f} images/s', log_file)
            start, seen = time.time(), 0


@torch.no_grad()
def evaluate(model, loader, gt_json):
    """Classification and COCO detection metrics on one split, and the detections in COCO result format."""
    model.eval()
    labels, predicted, detections, image_ids = [], [], [], []
    for images, y, targets in loader:
        with torch.autocast('cuda', dtype=torch.bfloat16):
            logits, outputs, _ = model(images.cuda())
        labels += y.tolist()
        predicted += logits.argmax(dim=1).tolist()
        for target, out in zip(targets, outputs):
            image_ids.append(target['image_id'])
            boxes = to_original_xywh(out['boxes'], target['orig_size']).tolist()
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
    backbone = build_backbone(args.backbone, pretrained=args.init == 'pretrained',
                              in_chans=len(args.input_channels.split(',')))
    if args.adapt != 'full':
        add_adapters(backbone, args.adapt, args.lora_rank)  # also freezes the backbone
        if args.train_patch_embed:
            backbone.patch_embed.requires_grad_(True)  # trained at --backbone-lr with the adapters
    model_class = SwinMultiTaskModel if args.backbone in SWIN_MODELS else MultiTaskModel
    model = model_class(backbone, task_routing=args.adapt == 'moelora', heatmap=args.heatmap_weight > 0,
                        heatmap_min_sigma=args.heatmap_min_sigma).cuda()
    parameters = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f'parameters: {parameters:,} total, {trainable:,} trainable', log_file)
    if parameters > MAX_PARAMETERS or trainable > MAX_TRAINABLE:
        log(f'WARNING: over the limits ({MAX_PARAMETERS / 1e6:.0f} M total, {MAX_TRAINABLE / 1e6:.0f} M trainable); '
            'not allowed for a submission', log_file)

    # Optimizer and schedule: trainable backbone parameters at --backbone-lr, all new modules at --lr
    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    head_params = [p for name, p in model.named_parameters() if not name.startswith('backbone.')]
    optimizer = torch.optim.AdamW([{'params': backbone_params, 'lr': args.backbone_lr}, {'params': head_params}],
                                  lr=args.lr, weight_decay=args.weight_decay)
    warmup, total = len(train_loader), args.epochs * len(train_loader)  # one epoch of warmup
    schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: lr_factor(step, warmup, total))

    # Train, validating every --eval-every epochs; the reported model is the last-epoch model
    history, start = [], time.time()
    for epoch in range(1, args.epochs + 1):
        train_one_epoch(model, train_loader, optimizer, schedule, epoch, log_file, args.heatmap_weight)
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            val, val_detections = evaluate(model, val_loader, args.data / 'detection' / 'instances_val.json')
            history.append({'epoch': epoch, **val})
            log(f'epoch {epoch} val: {headline(val)}', log_file)
    train_hours = (time.time() - start) / 3600

    # Evaluate on test, once
    test, test_detections = evaluate(model, test_loader, args.data / 'detection' / 'instances_test.json')
    peak_gb = torch.cuda.max_memory_allocated() / 2**30
    log(f'test: {headline(test)}\ntraining {train_hours:.2f} h, peak GPU memory {peak_gb:.1f} GB', log_file)

    # Save
    summary = {'run': run_dir.name, 'commit': commit, 'dirty': dirty,
               'args': vars(args), 'parameters': parameters, 'trainable_parameters': trainable,
               'train_hours': train_hours, 'peak_gpu_memory_gb': peak_gb,
               'val_history': history, 'val': history[-1], 'test': test}
    (run_dir / 'metrics.json').write_text(json.dumps(summary, indent=2, default=str))
    (run_dir / 'predictions_val.json').write_text(json.dumps(val_detections))
    (run_dir / 'predictions_test.json').write_text(json.dumps(test_detections))
    torch.save(model.state_dict(), run_dir / 'model.pt')
    append_results(summary, args.out / 'results.csv')


if __name__ == '__main__':
    main()
