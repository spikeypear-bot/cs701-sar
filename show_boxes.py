"""Draw ground-truth and predicted boxes of chosen classes on holdout images, one contact sheet per class.
Example: python show_boxes.py runs/swin_b-moelora-e36_20261017-034036 --classes bridge,playground
Green: ground truth. Red: predictions of that class (score shown), orange: of another class. Needs only Pillow."""
import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('run', type=Path, help='run folder of train.py (reads its predictions_test.json)')
    p.add_argument('--classes', default='bridge,playground', help='comma-separated class names')
    p.add_argument('--gt', type=Path, default=Path('data/detection/instances_test.json'))
    p.add_argument('--images', type=Path, default=Path('data'), help='folder the file_name entries are relative to')
    p.add_argument('--n', type=int, default=20, help='images per class')
    p.add_argument('--score', type=float, default=0.3, help='hide predictions below this score')
    p.add_argument('--tile', type=int, default=384, help='each image is resized to this many pixels per side')
    p.add_argument('--out', type=Path, help='where the sheets go (default: <run>/boxes)')
    return p.parse_args()


def draw(image, boxes, scale, colour, labels=None):
    """Boxes are xywh in original pixels; `labels` is printed above each box."""
    pen = ImageDraw.Draw(image)
    for k, (x, y, w, h) in enumerate(boxes):
        pen.rectangle([x * scale, y * scale, (x + w) * scale, (y + h) * scale], outline=colour, width=2)
        if labels:
            pen.text((x * scale, max(0, y * scale - 11)), labels[k], fill=colour)


def main():
    args = parse_args()
    gt = json.loads(args.gt.read_text())
    names = {c['id']: c['name'] for c in gt['categories']}
    ids = {name: i for i, name in names.items()}
    images = {im['id']: im for im in gt['images']}
    truth, predicted = {}, {}
    for a in gt['annotations']:
        truth.setdefault(a['image_id'], []).append(a)
    for d in json.loads((args.run / 'predictions_test.json').read_text()):
        if d['score'] >= args.score:
            predicted.setdefault(d['image_id'], []).append(d)
    out = args.out or args.run / 'boxes'
    out.mkdir(parents=True, exist_ok=True)

    for name in args.classes.split(','):
        chosen = sorted(i for i, boxes in truth.items() if any(b['category_id'] == ids[name] for b in boxes))[:args.n]
        columns = 5
        sheet = Image.new('RGB', (columns * args.tile, -(-len(chosen) // columns) * (args.tile + 14)), 'white')
        for k, image_id in enumerate(chosen):
            im = images[image_id]
            scale = args.tile / max(im['width'], im['height'])
            tile = Image.open(args.images / im['file_name']).convert('RGB')
            tile = tile.resize((round(im['width'] * scale), round(im['height'] * scale)))
            draw(tile, [b['bbox'] for b in truth[image_id]], scale, (0, 220, 0))
            same = [d for d in predicted.get(image_id, []) if d['category_id'] == ids[name]]
            other = [d for d in predicted.get(image_id, []) if d['category_id'] != ids[name]]
            draw(tile, [d['bbox'] for d in other], scale, (255, 150, 0),
                 [f"{names[d['category_id']]} {d['score']:.2f}" for d in other])
            draw(tile, [d['bbox'] for d in same], scale, (255, 0, 0), [f"{d['score']:.2f}" for d in same])
            x, y = k % columns * args.tile, k // columns * (args.tile + 14)
            sheet.paste(tile, (x, y + 14))
            ImageDraw.Draw(sheet).text((x + 2, y + 1), f"{im['file_name']} {im['width']}x{im['height']} "
                                                       f"gt {len(truth[image_id])}", fill='black')
        sheet.save(out / f'{name}.jpg', quality=90)
        print(f'{name}: {len(chosen)} images -> {out / f"{name}.jpg"}')


if __name__ == '__main__':
    main()
