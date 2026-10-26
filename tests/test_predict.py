"""predict.py's per-class box offsets, on hand-made predictions (no model, no GPU)."""
import argparse

import pytest

from predict import CLASS_NAMES, parse_offsets, remove_offsets


def test_parse_offsets_maps_class_names_to_labels():
    assert parse_offsets('wind_turbine=0.5:0.3,ship=0.4:-0.5') == {8: (0.5, 0.3), 7: (0.4, -0.5)}
    assert CLASS_NAMES.index('aircraft') == 0 and parse_offsets('') == {}
    with pytest.raises(argparse.ArgumentTypeError):
        parse_offsets('turbine=0.5:0.3')


def test_remove_offsets_moves_only_its_classes_by_original_pixels():
    images = [{'file_name': 'a.png', 'width': 256, 'height': 128}]  # input 512: x2, y4 per original pixel
    preds = {'a.png': (8, [[10., 10., 20., 20.], [30., 30., 40., 40.]], [8, 0], [0.9, 0.8])}
    label, boxes, classes, scores = remove_offsets(preds, images, {8: (0.5, 0.25)}, size=512)['a.png']
    assert boxes == [[9., 9., 19., 19.], [30., 30., 40., 40.]]  # 0.5 x 2 and 0.25 x 4 input pixels
    assert (label, classes, scores) == (8, [8, 0], [0.9, 0.8])
