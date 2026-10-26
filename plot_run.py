"""Over- or underfitting, and where the errors are, of train.py and yolo26.py runs: training curves and the holdout's
confusion matrix as PNGs in each run's plots/ folder, per-class AP of all the runs side by side, and a few curve
numbers printed per run. Works on a run still training (curves only). Needs no GPU (matplotlib, pycocotools).
Example: python plot_run.py runs/yolov8m-e100_20261019-222115 runs/y26x-p2-f5-8-e100_20261026-210000
Curves: train.py runs have no validation loss, so they show the mean training loss per epoch (from log.txt) next to
val mAP and macro-F1 (every --eval-every epochs); yolo26.py runs show Ultralytics' train and val losses and val mAP
per epoch (ultralytics/results.csv). Val here is the local holdout, the same images as test (make_split.py)."""
import argparse
import contextlib
import csv
import io
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')  # no display on the cluster
import matplotlib.pyplot as plt
import numpy as np
from pycocotools.coco import COCO

from analyze_runs import evaluate, mean_ap

ITER = re.compile(r'^epoch (\d+) iter \d+/\d+ loss ([\d.]+) \((.*)\) lr')
VAL = re.compile(r'^epoch (\d+) val: (.*)$')


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('runs', type=Path, nargs='+', help='run folders of train.py or yolo26.py')
    p.add_argument('--gt', type=Path, default=Path('data/detection/instances_test.json'),
                   help='the boxes of the holdout (default: the copy train.py scores as test)')
    p.add_argument('--out', type=Path, default=Path('runs/plots/class_ap.png'),
                   help='the per-class AP chart of all the runs (default: runs/plots/class_ap.png)')
    return p.parse_args()


def train_py_curves(run):
    """From log.txt: {loss name: (epochs, mean per epoch)} over the logged iterations, and {metric: (epochs, values)}
    of the val lines."""
    sums, counts = defaultdict(lambda: defaultdict(float)), defaultdict(lambda: defaultdict(int))
    val = defaultdict(lambda: ([], []))
    for line in (run / 'log.txt').read_text().splitlines():
        if m := ITER.match(line):
            epoch = int(m[1])
            parts = m[3].split()
            for name, value in [('total', m[2]), *zip(parts[::2], parts[1::2])]:
                sums[name][epoch] += float(value)
                counts[name][epoch] += 1
        elif m := VAL.match(line):
            words = m[2].split()
            for name, value in zip(words[::2], words[1::2]):
                val[name][0].append(int(m[1]))
                val[name][1].append(float(value))
    losses = {name: (sorted(s), [s[e] / counts[name][e] for e in sorted(s)]) for name, s in sums.items()}
    return losses, dict(val)


def ultralytics_curves(run):
    """From ultralytics/results.csv: {loss name: {'train': (epochs, values), 'val': ...}} and {metric: (epochs,
    values)} of the metrics/ columns."""
    with open(run / 'ultralytics' / 'results.csv') as f:
        rows = [{k.strip(): v for k, v in row.items()} for row in csv.DictReader(f)]
    epochs = [int(float(r['epoch'])) for r in rows]
    losses, metrics = defaultdict(dict), {}
    for column in rows[0]:
        values = [float(r[column]) for r in rows]
        if column.startswith(('train/', 'val/')):
            split, name = column.split('/', 1)
            losses[name][split] = (epochs, values)
        elif column.startswith('metrics/'):
            metrics[column.removeprefix('metrics/')] = (epochs, values)
    return dict(losses), metrics


def report(name, epochs, values, higher_better=True):
    """Peak epoch, last value and the change over the last quarter of the epochs, as one line."""
    values = np.array(values)
    best = int(np.argmax(values) if higher_better else np.argmin(values))
    quarter = [v for e, v in zip(epochs, values) if e >= epochs[-1] * 0.75]
    return (f'  {name}: best {values[best]:.4f} at epoch {epochs[best]}, last {values[-1]:.4f} at epoch {epochs[-1]}'
            f' ({values[-1] - values[best]:+.4f} vs best), last quarter {quarter[-1] - quarter[0]:+.4f}')


def plot_curves(run, path):
    """curves.png: losses on the left, val metrics on the right; prints the curve numbers."""
    fig, (left, right) = plt.subplots(1, 2, figsize=(13, 4.5))
    lines = []
    if (run / 'ultralytics' / 'results.csv').exists():
        losses, metrics = ultralytics_curves(run)
        for k, (name, splits) in enumerate(sorted(losses.items())):
            for split, style in (('train', '-'), ('val', '--')):
                if split in splits:
                    left.plot(*splits[split], style, color=f'C{k}', label=f'{split} {name}')
            if 'val' in splits:
                lines.append(report(f'val {name}', *splits['val'], higher_better=False))
        shown = {k: v for k, v in metrics.items() if k.startswith('mAP')}
        left.set_title('losses (solid: train, dashed: val)')
    else:
        losses, metrics = train_py_curves(run)
        for k, (name, (epochs, values)) in enumerate(sorted(losses.items())):
            left.plot(epochs, values, color=f'C{k}', lw=2.5 if name == 'total' else 1.2, label=name)
        lines.append(report('train loss total', *losses['total'], higher_better=False))
        shown = {k: v for k, v in metrics.items() if k in ('mAP', 'AP50', 'macro_f1')}
        left.set_title('training loss, mean per epoch (no val loss in train.py)')
    for name, (epochs, values) in shown.items():
        right.plot(epochs, values, 'o-', ms=3, label=f'val {name}')
        lines.append(report(f'val {name}', epochs, values))
    left.set_xlabel('epoch')
    left.set_yscale('log')
    left.legend(fontsize=7)
    right.set_xlabel('epoch')
    right.set_title('validation (local holdout)')
    right.grid(alpha=0.3)
    right.legend(fontsize=8)
    fig.suptitle(run.name)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return lines


def plot_confusion(cm, names, path, title):
    """confusion.png: rows = truth, columns = prediction, colour = share of the row, text = image count."""
    cm = np.array(cm)
    share = cm / cm.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(share, cmap='Blues', vmin=0, vmax=1)
    for t in range(len(names)):
        for p in range(len(names)):
            if cm[t, p]:
                ax.text(p, t, cm[t, p], ha='center', va='center', fontsize=8,
                        color='white' if share[t, p] > 0.5 else 'black')
    ax.set_xticks(range(len(names)), names, rotation=45, ha='right')
    ax.set_yticks(range(len(names)), names)
    ax.set_xlabel('predicted')
    ax.set_ylabel('truth')
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_class_ap(columns, names, path):
    """One group of bars per class (and 'all'), one bar per run."""
    labels = ['all', *names]
    width = 0.8 / len(columns)
    fig, ax = plt.subplots(figsize=(max(8, 1.2 * len(labels)), 4.5))
    for k, (run, aps) in enumerate(columns.items()):
        x = np.arange(len(labels)) + (k - (len(columns) - 1) / 2) * width
        ax.bar(x, [100 * a for a in aps], width, label=run)
    ax.set_xticks(range(len(labels)), labels, rotation=30, ha='right')
    ax.set_ylabel('AP (IoU 0.5:0.95), %')
    ax.set_title('holdout AP per class')
    ax.grid(axis='y', alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    args = parse_args()
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(args.gt)
    cat_ids = sorted(gt.getCatIds())
    names = [gt.cats[c]['name'] for c in cat_ids]
    image_ids = sorted(gt.getImgIds())
    class_ap = {}
    for run in args.runs:
        plots = run / 'plots'
        plots.mkdir(exist_ok=True)
        print(f'\n{run.name}')
        print('\n'.join(plot_curves(run, plots / 'curves.png')))
        if not (run / 'metrics.json').exists():
            print('  no metrics.json yet (still training?): curves only')
            continue
        test = json.loads((run / 'metrics.json').read_text())['test']
        plot_confusion(test['confusion_matrix'], names, plots / 'confusion.png',
                       f'{run.name}\nmacro-F1 {100 * test["macro_f1"]:.1f}')
        precision = evaluate(gt, json.loads((run / 'predictions_test.json').read_text()), image_ids)
        class_ap[run.name] = [mean_ap(precision[..., 0]), *(mean_ap(precision[:, :, k, 0]) for k in range(len(names)))]
        print(f'  holdout: mAP {100 * class_ap[run.name][0]:.1f}, macro-F1 {100 * test["macro_f1"]:.1f} -> {plots}')
    if class_ap:
        plot_class_ap(class_ap, names, args.out)
        print(f'\nper-class AP of {len(class_ap)} run(s) -> {args.out}')


if __name__ == '__main__':
    main()
