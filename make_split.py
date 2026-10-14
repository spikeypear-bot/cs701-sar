"""Split the labelled course train set into train + a local holdout, in the folder layout of sarbench/data.py.

The course val and test sets have no labels, so the holdout is our local val; test is a copy of it so train.py
runs unchanged. The split is stratified by class and fixed by --seed.
"""
import argparse
import csv
import json
import os
import random
from collections import Counter, defaultdict
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--src', type=Path, default=Path('cs701-sar-course-data/train'))
    p.add_argument('--out', type=Path, default=Path('data'))
    p.add_argument('--holdout', type=float, default=0.1, help='fraction of every class held out')
    p.add_argument('--seed', type=int, default=0)
    return p.parse_args()


def stratified_split(rows, holdout, seed):
    """The same fraction of every class goes to the holdout."""
    rng = random.Random(seed)
    by_class = defaultdict(list)
    for row in rows:
        by_class[row['label']].append(row)
    train, hold = [], []
    for label in sorted(by_class, key=int):  # fixed order, so the split depends only on the seed
        group = by_class[label]
        rng.shuffle(group)
        n_hold = round(len(group) * holdout)
        hold += group[:n_hold]
        train += group[n_hold:]
    return train, hold


def write_csv(rows, path):
    """Columns image, label: the names sarbench/data.py reads."""
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, ['image', 'label'])
        writer.writeheader()
        writer.writerows({'image': r['file_name'], 'label': r['label']} for r in rows)


def subset_coco(coco, image_ids):
    """The images of `image_ids` and their boxes; categories unchanged."""
    images = [img for img in coco['images'] if img['id'] in image_ids]
    assert len(images) == len(image_ids), 'some image_id of labels.csv is not in instances.json'
    annotations = [ann for ann in coco['annotations'] if ann['image_id'] in image_ids]
    return {'images': images, 'annotations': annotations, 'categories': coco['categories']}


def main():
    args = parse_args()
    with open(args.src / 'labels.csv') as f:
        rows = list(csv.DictReader(f))
    coco = json.loads((args.src / 'instances.json').read_text())
    train, hold = stratified_split(rows, args.holdout, args.seed)
    splits = {'train': train, 'val': hold, 'test': hold}  # test = val: no labelled test set

    (args.out / 'classification').mkdir(parents=True, exist_ok=True)
    (args.out / 'detection').mkdir(parents=True, exist_ok=True)
    for name, split_rows in splits.items():
        write_csv(split_rows, args.out / 'classification' / f'{name}.csv')
        ids = {int(r['image_id']) for r in split_rows}
        (args.out / 'detection' / f'instances_{name}.json').write_text(json.dumps(subset_coco(coco, ids)))

    # Paths in the CSVs are images/xxxxx.jpg relative to --out: link instead of copying
    link = args.out / 'images'
    if not link.exists():
        os.symlink((args.src / 'images').resolve(), link)

    for name in ('train', 'val'):
        counts = Counter(int(r['label']) for r in splits[name])
        print(f'{name}: {len(splits[name])} images, per class {[counts[c] for c in range(9)]}')
    assert len(train) + len(hold) == len(rows) and not {r['image_id'] for r in train} & {r['image_id'] for r in hold}


if __name__ == '__main__':
    main()
