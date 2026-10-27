"""yolo26.py's loss patches (--o2m-final, --stal-min) on Ultralytics' own classes; needs the cs701y26 env (8.4.189),
no GPU."""
from types import SimpleNamespace

import pytest

loss = pytest.importorskip('ultralytics.utils.loss')
if not hasattr(loss, 'E2ELoss'):
    pytest.skip('needs the cs701y26 env (Ultralytics with YOLO26)', allow_module_level=True)
from ultralytics.utils import tal

from yolo26 import patch_losses


@pytest.fixture(autouse=True)
def restore_classes(monkeypatch):
    monkeypatch.setattr(loss.E2ELoss, '__init__', loss.E2ELoss.__init__)
    monkeypatch.setattr(tal.TaskAlignedAssigner, '__init__', tal.TaskAlignedAssigner.__init__)


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
