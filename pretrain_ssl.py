"""SimMIM self-supervised pretraining of a Swin backbone's LoRA adapters on the training images (no labels); see
sarbench/simmim.py. Writes <run>/ssl.pt for train.py --ssl-weights. It checkpoints every epoch and stops before
--max-hours; continue with --resume <run>.
Example: python pretrain_ssl.py --backbone swin_b --rank 64 --unfreeze-blocks 23 --epochs 100"""
import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from sarbench.backbones import SWIN_MODELS, build_backbone
from sarbench.data import SARPretrain
from sarbench.simmim import SimMIM, add_ssl_adapters, ssl_checkpoint
from train import MAX_TRAINABLE, log, lr_factor, parse_blocks, save_code_version


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, default=Path(__file__).resolve().parents[1] / 'dataset' / 'SARFact-Course-20K')
    p.add_argument('--backbone', choices=list(SWIN_MODELS), default='swin_b')
    p.add_argument('--rank', type=int, default=64, help='of the LoRA adapters on every block, alpha = 2 x rank '
                                                        '(default: 64, as ExPLoRA; Swin-B: 72,192 parameters per rank)')
    p.add_argument('--unfreeze-blocks', type=parse_blocks, default=[],
                   help='also fully train these blocks, e.g. 23 (ExPLoRA unfreezes the last one; Swin-B: about 0.2, '
                        '0.8, 3.2 and 12.6 M parameters per block of stages 1-4)')
    p.add_argument('--mask-ratio', type=float, default=0.6, help='share of the 32 x 32 squares hidden (default: 0.6)')
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--warmup-epochs', type=int, default=10, help='default: 10, as SimMIM')
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--lr', type=float, default=1e-4, help='adapters, mask token and decoder (default: 1e-4, as '
                                                         "train.py's adapters)")
    p.add_argument('--unfreeze-lr', type=float, default=2e-5, help='pretrained weights of --unfreeze-blocks')
    p.add_argument('--weight-decay', type=float, default=0.05)
    p.add_argument('--eval-every', type=int, default=10, help='holdout loss every N epochs, and after the last one')
    p.add_argument('--max-hours', type=float, default=5.5,
                   help='stop after the last epoch that ends within this time (train.slurm allows 6 h)')
    p.add_argument('--resume', type=Path, help='a run folder of this script: continue it, with its own options')
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--limit', type=int, help='use only the first N images of every split (quick checks)')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out', type=Path, default=Path('runs'))
    p.add_argument('--name', help='run name; the run folder is <out>/<name>_<time> (default: <backbone>-simmim)')
    args = p.parse_args()
    if args.resume:  # the run's options; only how long this job may take, and its loaders' workers, are new
        saved = json.loads((args.resume / 'args.json').read_text())
        args = argparse.Namespace(**{**saved, 'data': Path(saved['data']), 'out': Path(saved['out']),
                                     'resume': args.resume, 'max_hours': args.max_hours, 'workers': args.workers})
    return args


@torch.no_grad()
def holdout_loss(model, loader):
    """Mean SimMIM loss on the holdout, with the same masks every time (fixed seed), so values are comparable."""
    model.eval()
    generator = torch.Generator(device='cuda').manual_seed(0)
    total, count = 0.0, 0
    for images in loader:
        images = images.cuda()
        masks = model.random_masks(len(images), *images.shape[-2:], generator=generator)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            total += model(images, masks).item() * len(images)
        count += len(images)
    return total / count


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    if args.resume:
        run_dir = args.resume
    else:
        run_dir = args.out / f'{args.name or args.backbone + "-simmim"}_{time.strftime("%Y%m%d-%H%M%S")}'
        run_dir.mkdir(parents=True)
        (run_dir / 'args.json').write_text(json.dumps(vars(args), default=str))
    log_file = open(run_dir / 'log.txt', 'a')
    log(json.dumps(vars(args), default=str), log_file)
    commit, dirty = save_code_version(run_dir)
    log(f'code: commit {commit}' + (', with uncommitted changes (code.diff)' if dirty else ''), log_file)

    loader = lambda split, train: DataLoader(SARPretrain(args.data, split, train=train, limit=args.limit),
                                             args.batch_size, shuffle=train, num_workers=args.workers, drop_last=train,
                                             persistent_workers=train and args.workers > 0)
    train_loader, holdout_loader = loader('train', True), loader('val', False)  # val: the local holdout

    # Model: ImageNet Swin, frozen; LoRA on every block and --unfreeze-blocks train, with the mask token and decoder
    backbone = build_backbone(args.backbone, pretrained=True)
    unfrozen = {id(p) for p in add_ssl_adapters(backbone, args.rank, args.unfreeze_blocks)}  # at --unfreeze-lr
    model = SimMIM(backbone, args.mask_ratio).cuda()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f'parameters: {sum(p.numel() for p in model.parameters()):,} total, {trainable:,} trainable', log_file)
    if trainable > MAX_TRAINABLE:
        log(f'WARNING: over the {MAX_TRAINABLE / 1e6:.0f} M trainable limit before fine-tuning adds its own', log_file)
    new = [p for p in model.parameters() if p.requires_grad and id(p) not in unfrozen]  # .cuda() keeps the objects
    groups = [{'params': new}]
    if unfrozen:
        groups.append({'params': [p for p in model.parameters() if id(p) in unfrozen], 'lr': args.unfreeze_lr})
    optimizer = torch.optim.AdamW(groups, lr=args.lr, weight_decay=args.weight_decay)
    warmup, total = args.warmup_epochs * len(train_loader), args.epochs * len(train_loader)
    schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: lr_factor(step, warmup, total))

    # Resume: the trainable weights, optimizer and schedule of the last finished epoch
    first_epoch, history = 1, []
    if args.resume:
        state = torch.load(run_dir / 'checkpoint.pt', map_location='cpu')
        model.load_state_dict(state['model'], strict=False)  # the frozen ImageNet weights are not in it
        optimizer.load_state_dict(state['optimizer'])
        schedule.load_state_dict(state['schedule'])
        first_epoch, history = state['epoch'] + 1, state['history']
        log(f'resumed after epoch {state["epoch"]}', log_file)

    torch.manual_seed(args.seed + first_epoch - 1)  # a resumed run does not repeat its first epochs' order and masks
    start = time.time()
    for epoch in range(first_epoch, args.epochs + 1):
        epoch_start = time.time()
        model.train()
        window, seen = time.time(), 0
        for step, images in enumerate(train_loader, 1):
            images = images.cuda()
            masks = model.random_masks(len(images), *images.shape[-2:])
            with torch.autocast('cuda', dtype=torch.bfloat16):
                loss = model(images, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            schedule.step()
            seen += len(images)
            if step % 50 == 0 or step == len(train_loader):
                log(f'epoch {epoch} iter {step}/{len(train_loader)} loss {loss.item():.4f} '
                    f'lr {optimizer.param_groups[0]["lr"]:.2e} {seen / (time.time() - window):.1f} images/s', log_file)
                window, seen = time.time(), 0
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            history.append({'epoch': epoch, 'holdout_loss': holdout_loss(model, holdout_loader)})
            log(f'epoch {epoch} holdout loss {history[-1]["holdout_loss"]:.4f}', log_file)
        trained = {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad}
        torch.save({'model': trained, 'optimizer': optimizer.state_dict(), 'schedule': schedule.state_dict(),
                    'epoch': epoch, 'history': history}, run_dir / 'checkpoint.pt')
        hours, epoch_hours = (time.time() - start) / 3600, (time.time() - epoch_start) / 3600
        if epoch < args.epochs and hours + epoch_hours > args.max_hours:
            log(f'stopping after epoch {epoch} ({hours:.2f} h); continue with: --resume {run_dir}', log_file)
            return

    checkpoint = ssl_checkpoint(model, args.backbone, args.rank, args.unfreeze_blocks)
    torch.save(checkpoint, run_dir / 'ssl.pt')
    (run_dir / 'metrics.json').write_text(json.dumps(
        {'run': run_dir.name, 'commit': commit, 'dirty': dirty, 'args': vars(args), 'history': history,
         'trained_parameters': checkpoint['trained_parameters']}, indent=2, default=str))
    log(f'saved {run_dir / "ssl.pt"}: {checkpoint["trained_parameters"]:,} parameters trained, '
        f'{checkpoint["pretraining_only_parameters"]:,} of them pretraining-only', log_file)


if __name__ == '__main__':
    main()
