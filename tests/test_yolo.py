"""sarbench/yolo.py on real images: losses and outputs in model.py's format, the COCO weights on a 1-channel input, a
model.pt that predict.py can rebuild, and an overfit check (a pretrained YOLOv8m fits 16 training images)."""
import json

import torch
import torch.nn.functional as F

from predict import load_model
from sarbench.data import to_original_xywh
from sarbench.metrics import detection_metrics
from sarbench.yolo import YOLOMultiTaskModel, pretrained_state
from test_model import ROOT, first_train_images, forward  # pytest puts tests/ on the path


def test_training_forward_returns_logits_and_the_three_yolo_losses():
    model = YOLOMultiTaskModel('yolov8m', pretrained=False).cuda().train()
    images, _, targets = first_train_images(2)
    logits, detections, losses = forward(model, images, targets)
    assert logits.shape == (2, 9) and detections == []
    assert set(losses) == {'loss_yolo_box', 'loss_yolo_cls', 'loss_yolo_dfl'}
    assert all(loss.dtype == torch.float32 and loss.isfinite() and loss > 0 for loss in losses.values())


def test_detections_are_fp32_unquantized_and_labelled_1_to_9():
    model = YOLOMultiTaskModel('yolov8m', pretrained=True).cuda().eval()
    with torch.no_grad():
        logits, detections, losses = forward(model, first_train_images(2)[0])
    assert losses == {} and len(detections) == 2
    boxes, labels = torch.cat([d['boxes'] for d in detections]), torch.cat([d['labels'] for d in detections])
    assert all(0 < len(d['boxes']) <= 100 for d in detections)
    assert boxes.dtype == torch.float32 and detections[0]['scores'].dtype == torch.float32
    assert boxes.min() >= 0 and boxes.max() <= 512 and labels.min() >= 1 and labels.max() <= 9
    assert (boxes != boxes.bfloat16().float()).any()  # not every coordinate is on the bf16 grid


def test_coco_weights_load_on_one_channel_except_the_new_class_convs():
    model = YOLOMultiTaskModel('yolov8m', pretrained=True)
    own = model.backbone.state_dict()
    assert model.transferred == len(own) - 6  # Detect's last class conv, weight and bias, on each of 3 levels
    rgb = pretrained_state('yolov8m', in_chans=3)['model.0.conv.weight']
    assert torch.allclose(own['model.0.conv.weight'], rgb.sum(dim=1, keepdim=True))
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) < 40e6


def test_yolov8l_with_6_frozen_layers_is_under_the_trainable_limit():
    model = YOLOMultiTaskModel('yolov8l', pretrained=False, freeze_layers=6)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert 39.6e6 < trainable < 40e6
    assert not any(p.requires_grad for p in model.backbone.model[:6].parameters())
    assert all(p.requires_grad for p in model.backbone.model[6].parameters())


def test_predict_rebuilds_the_model_from_a_run_folder(tmp_path):
    model = YOLOMultiTaskModel('yolov8m', pretrained=False).cuda().eval()
    torch.save(model.state_dict(), tmp_path / 'model.pt')
    (tmp_path / 'metrics.json').write_text(json.dumps({'args': {'backbone': 'yolov8m', 'adapt': 'full'}}))
    images = first_train_images(2)[0]
    with torch.no_grad():
        expected, rebuilt = forward(model, images), forward(load_model(tmp_path), images)
    assert torch.allclose(expected[0], rebuilt[0])
    assert all(torch.allclose(a['boxes'], b['boxes']) for a, b in zip(expected[1], rebuilt[1]))


def test_pretrained_yolov8m_overfits_16_training_images():
    torch.manual_seed(0)
    images, labels, targets = first_train_images(16)
    model = YOLOMultiTaskModel('yolov8m', pretrained=True).cuda().train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=5e-4)
    for _ in range(200):
        with torch.autocast('cuda', dtype=torch.bfloat16):
            logits, _, losses = model(images, targets)
            loss = F.cross_entropy(logits, labels) + sum(losses.values())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        logits, detections, _ = forward(model.eval(), images)
    assert (logits.argmax(dim=1) == labels).all()
    predictions = []
    for t, d in zip(targets, detections):
        boxes = to_original_xywh(d['boxes'], t['orig_size']).tolist()
        predictions += [{'image_id': t['image_id'], 'category_id': c, 'bbox': b, 'score': s}
                        for b, c, s in zip(boxes, d['labels'].tolist(), d['scores'].tolist())]
    image_ids = [t['image_id'] for t in targets]
    assert detection_metrics(ROOT / 'detection' / 'instances_train.json', predictions, image_ids)['AP50'] > 0.9
