"""yolo26.py's patches (--o2m-final, --stal-min, --ignore-dim, --class-scale, --cls-weights) and --jpeg on Ultralytics' own classes; needs the cs701y26 env (8.4.189),
no GPU."""
from types import SimpleNamespace

import pytest

loss = pytest.importorskip('ultralytics.utils.loss')
if not hasattr(loss, 'E2ELoss'):
    pytest.skip('needs the cs701y26 env (Ultralytics with YOLO26)', allow_module_level=True)
from ultralytics.utils import tal

from yolo26 import albumentations, patch_class_scale, patch_losses, set_class_weights


@pytest.fixture(autouse=True)
def restore_classes(monkeypatch):
    monkeypatch.setattr(loss.E2ELoss, '__init__', loss.E2ELoss.__init__)
    monkeypatch.setattr(tal.TaskAlignedAssigner, '__init__', tal.TaskAlignedAssigner.__init__)
    monkeypatch.setattr(loss.v8DetectionLoss, 'get_assigned_targets_and_loss',
                        loss.v8DetectionLoss.get_assigned_targets_and_loss)


def e2e_loss(epochs=100):
    """An E2ELoss whose one-to-many / one-to-one parts are stubs: only its weight schedule is real."""
    return loss.E2ELoss(None, loss_fn=lambda model, **kwargs: SimpleNamespace(hyp=SimpleNamespace(epochs=epochs)))


def test_ultralytics_defaults_unchanged_without_flags():
    patch_losses()
    assert e2e_loss().decay(99) == pytest.approx(0.1)
    assert tal.TaskAlignedAssigner(stride=[4, 8, 16, 32]).stride_val == 8


def test_o2m_final_keeps_the_one_to_many_weight():
    patch_losses(o2m_final=0.8)
    criterion = e2e_loss()
    assert criterion.decay(0) == pytest.approx(0.8) and criterion.decay(99) == pytest.approx(0.8)


def test_stal_min_sets_the_tiny_box_floor():
    patch_losses(stal_min=16)
    assert tal.TaskAlignedAssigner(stride=[4, 8, 16, 32]).stride_val == 16


def test_ignored_boxes_get_no_loss():
    """An ignored box (class 9) is no target, and scores at anchors inside it get no gradient; a real box still
    trains, and so does the background elsewhere."""
    import torch
    from ultralytics.nn.tasks import DetectionModel
    patch_losses(ignore_cls=9)
    model = DetectionModel('yolo26n.yaml', nc=9, verbose=False)
    model.args = SimpleNamespace(box=7.5, cls=0.5, dfl=1.5)
    criterion = loss.v8DetectionLoss(model)
    preds = model.train()(torch.rand(1, 3, 128, 128))['one2many']
    preds['scores'].retain_grad()
    batch = {'batch_idx': torch.zeros(2), 'cls': torch.tensor([[0.], [9.]]),
             'bboxes': torch.tensor([[0.25, 0.25, 0.2, 0.2], [0.75, 0.75, 0.25, 0.25]])}  # normalised cx, cy, w, h
    total, _ = criterion.loss(preds, batch)
    total.sum().backward()
    anchors, strides = tal.make_anchors(preds['feats'], criterion.stride, 0.5)
    points = anchors * strides
    in_ignored = ((points > 80) & (points < 112)).all(-1)  # the class-9 box: 80-112 px on a 128 px input
    grad = preds['scores'].grad[0]  # (classes, anchors)
    assert in_ignored.any() and grad[:, in_ignored].abs().max() == 0
    assert grad[:, ~in_ignored].abs().max() > 0


def test_class_scale_resizes_training_images_by_class():
    import numpy as np
    from ultralytics.data.base import BaseDataset
    original = BaseDataset.load_image
    try:
        BaseDataset.load_image = lambda self, i: (np.zeros((512, 512, 3), np.uint8), (256, 256), (512, 512))
        patch_class_scale({0: 0.5, 3: 0.5}, default=0.0)
        data = SimpleNamespace(augment=True, labels=[{'cls': np.array([[0.]])}, {'cls': np.array([[1.]])},
                                                     {'cls': np.array([[3.], [9.]])}])  # 9: an --ignore-dim box
        for i in (0, 2):
            shapes = {BaseDataset.load_image(data, i)[2] for _ in range(50)}
            assert len(shapes) > 1 and all(h == w and 256 <= h <= 768 for h, w in shapes)
        assert BaseDataset.load_image(data, 1)[2] == (512, 512)  # default 0: unchanged
        data.augment = False  # validation
        assert BaseDataset.load_image(data, 0)[2] == (512, 512)
        assert BaseDataset.load_image(data, 0)[1] == (256, 256)
    finally:
        BaseDataset.load_image = original


def test_jpeg_builds_with_the_installed_albumentations():
    transforms = albumentations(SimpleNamespace(mult_noise=0, gamma=0, brightness_contrast=0, jpeg=50, albu_p=0.5))
    assert len(transforms) == 1


def test_cls_weights_set_on_the_trained_model():
    import inspect
    import torch
    assert 'class_weights' in inspect.getsource(loss.v8DetectionLoss.__init__)  # the hook the loss reads
    added = {}
    set_class_weights(SimpleNamespace(add_callback=lambda event, f: added.setdefault(event, f)), [1.0, 2.0, 1.0])
    trainer = SimpleNamespace(model=torch.nn.Linear(1, 1), device='cpu')
    added['on_pretrain_routine_end'](trainer)
    assert trainer.model.class_weights.tolist() == [1.0, 2.0, 1.0]
