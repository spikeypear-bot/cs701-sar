"""sarbench/yolo.py on real images: losses and outputs in model.py's format, the COCO weights on a 1-channel input, a
model.pt that predict.py can rebuild, and an overfit check (a pretrained YOLOv8m fits 16 training images); the same
for the P2 model (yolov8m-p2)."""
import json
import math

import pytest
import torch
import torch.nn.functional as F

from predict import load_model
from sarbench.data import to_original_xywh
from sarbench.metrics import detection_metrics
from sarbench.quadfpn import QUAD_FPN_PARTS
from sarbench.yolo import YOLOMultiTaskModel, enlarged_candidates, pretrained_state, soft_nms
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


def test_quad_fpn_parts_sit_in_the_neck_of_yolov8_and_yolo11():
    for name, laterals, upsamples, downsamples in (('yolov8m', ['4', '6', '9'], ['10', '13'], ['16', '19']),
                                                   ('yolo11m', ['4', '6', '10'], ['11', '14'], ['17', '20'])):
        quad = YOLOMultiTaskModel(name, pretrained=False, quad_fpn=QUAD_FPN_PARTS).quad_fpn
        assert (list(quad.deform), list(quad.upsample), list(quad.attend)) == (laterals, upsamples, downsamples)
        assert len(quad.balance.into) == 3


def test_quad_fpn_starts_as_the_plain_yolov8m_and_is_under_the_trainable_limit():
    torch.manual_seed(0)
    plain = YOLOMultiTaskModel('yolov8m', pretrained=False).cuda().eval()
    quad = YOLOMultiTaskModel('yolov8m', pretrained=False, quad_fpn=QUAD_FPN_PARTS).cuda().eval()
    missing, unexpected = quad.load_state_dict(plain.state_dict(), strict=False)
    assert not unexpected and missing and all(k.startswith('quad_fpn.') for k in missing)
    images, _, targets = first_train_images(2)
    with torch.no_grad():
        expected, got = forward(plain, images), forward(quad, images)
    assert torch.allclose(expected[0], got[0])
    assert all(torch.allclose(a['boxes'], b['boxes']) for a, b in zip(expected[1], got[1]))
    added = sum(p.numel() for p in quad.quad_fpn.parameters())
    assert 11e6 < added < 12e6 and sum(p.numel() for p in quad.parameters()) < 40e6
    quad.train()
    _, _, losses = forward(quad, images, targets)
    sum(losses.values()).backward()
    assert all(p.grad is not None for p in quad.quad_fpn.parameters())  # every new module is in the graph


def test_p2_model_adds_a_stride_4_level_and_takes_the_coco_weights_of_the_layers_it_shares():
    model = YOLOMultiTaskModel('yolov8m-p2', pretrained=True)
    head = model.backbone.model[-1]
    assert head.f == [18, 21, 24, 27] and head.stride.tolist() == [4, 8, 16, 32]
    own, coco = model.backbone.state_dict(), pretrained_state('yolov8m', in_chans=1)
    shared = [k for k in own if int(k.split('.')[1]) < 16]  # backbone and top-down neck: as in yolov8m
    assert model.transferred == len(shared) and all(torch.equal(own[k], coco[k]) for k in shared)
    assert 25.0e6 < sum(p.numel() for p in model.parameters()) < 25.1e6  # 25.06 M by hand from yolov8-p2.yaml
    with pytest.raises(ValueError):
        YOLOMultiTaskModel('yolov8m-p2', pretrained=False, centre_keypoint=True)


def test_p2_model_trains_and_detects_on_all_four_levels():
    model = YOLOMultiTaskModel('yolov8m-p2', pretrained=True).cuda().train()
    images, _, targets = first_train_images(2)
    _, _, losses = forward(model, images, targets)
    assert set(losses) == {'loss_yolo_box', 'loss_yolo_cls', 'loss_yolo_dfl'}
    assert all(loss.dtype == torch.float32 and loss.isfinite() and loss > 0 for loss in losses.values())
    sum(losses.values()).backward()
    new = [p for p in model.backbone.model[16:].parameters() if p.requires_grad]  # the untrained P2 neck and head
    assert new and all(p.grad is not None for p in new)  # all in the graph
    # the P2 class branch is trained at every stride-4 point (its box branch only where a box is assigned)
    assert all(p.grad.abs().sum() > 0 for p in model.backbone.model[-1].cv3[0].parameters())
    model.eval()
    with torch.no_grad():
        _, detections, _ = forward(model, images)
    assert model.backbone.model[-1].anchors.shape[1] == 128**2 + 64**2 + 32**2 + 16**2  # points at strides 4-32
    boxes = torch.cat([d['boxes'] for d in detections])
    assert all(0 < len(d['boxes']) <= 100 for d in detections) and boxes.min() >= 0 and boxes.max() <= 512


def test_soft_nms_ranks_an_overlapping_box_of_the_same_class_lower_instead_of_dropping_it():
    # 3 anchors, 2 classes: boxes 0 and 1 overlap at IoU 360 / 440 (class 0); box 2 is box 1 again, but class 1
    boxes = torch.tensor([[50., 50, 20, 20], [52, 50, 20, 20], [52, 50, 20, 20]])  # centre-xywh
    scores = torch.tensor([[0.9, 0.0], [0.8, 0.0], [0.0, 0.7]])
    kept = soft_nms(torch.cat([boxes, scores], dim=1).T[None], score_thresh=0.001, sigma=0.5, max_det=100, nc=2)[0]
    assert kept[:, 5].tolist() == [0, 1, 0]
    assert torch.allclose(kept[:, 4], torch.tensor([0.9, 0.7, 0.8 * math.exp(-(360 / 440) ** 2 / 0.5)]))
    assert torch.allclose(kept[0, :4], torch.tensor([40., 40, 60, 60]))
    assert soft_nms(torch.cat([boxes, scores], dim=1).T[None], 0.001, 0.5, max_det=2, nc=2)[0].shape == (2, 6)
    keypoints = torch.tensor([[51., 49], [53, 50], [52, 51]])  # extra columns are carried along
    kept = soft_nms(torch.cat([boxes, scores, keypoints], dim=1).T[None], 0.001, 0.5, max_det=100, nc=2)[0]
    assert kept[:, 6:].tolist() == [[51, 49], [52, 51], [53, 50]]


def test_centre_keypoint_model_trains_on_box_centres_and_can_move_boxes_onto_them():
    model = YOLOMultiTaskModel('yolov8m', pretrained=True, centre_keypoint=True).cuda().train()
    own = model.backbone.state_dict()
    # untrained: the class convs (9 classes) and the keypoint branch cv4
    assert all(k.startswith('model.22.cv4.') or k.startswith('model.22.cv3.') for k in own if k not in
               {k for k, v in pretrained_state('yolov8m', in_chans=1).items() if k in own and v.shape == own[k].shape})
    images, _, targets = first_train_images(2)
    _, _, losses = forward(model, images, targets)
    assert set(losses) == {'loss_yolo_box', 'loss_yolo_cls', 'loss_yolo_dfl', 'loss_yolo_kpt'}
    assert all(loss.dtype == torch.float32 and loss.isfinite() and loss > 0 for loss in losses.values())
    model.eval()
    with torch.no_grad():
        plain = forward(model, images)[1]
        model.keypoint_centres = True
        moved = forward(model, images)[1]
    for a, b in zip(plain, moved):  # same detections, other places (sizes equal unless clipped)
        assert torch.allclose(a['scores'], b['scores']) and torch.equal(a['labels'], b['labels'])
        assert not torch.equal(a['boxes'], b['boxes'])


def test_small_box_assign_gives_candidates_to_a_box_between_the_grid_points():
    from ultralytics.utils.tal import TaskAlignedAssigner
    select = TaskAlignedAssigner.select_candidates_in_gts
    grid = torch.arange(4., 32, 8)  # stride-8 point centres 4, 12, 20, 28
    points = torch.stack(torch.meshgrid(grid, grid, indexing='xy'), dim=-1).view(-1, 2)
    boxes = torch.tensor([[[5., 5, 11, 11], [0, 0, 0, 0], [2, 2, 30, 30]]])  # 6 px between points, padding, large
    plain, enlarged = select(points, boxes), enlarged_candidates(16, select)(points, boxes)
    assert plain[0, 0].sum() == 0 and enlarged[0, 0].sum() == 4  # widened to [0, 16]: points 4 and 12 in x and y
    assert enlarged[0, 1].sum() == 0  # the padding stays empty
    assert plain[0, 2].sum() == 16 and torch.equal(plain[0, 2], enlarged[0, 2])  # a large box is unchanged


def test_small_box_assign_trains_and_patches_only_its_own_assigner():
    images, _, targets = first_train_images(2)
    model = YOLOMultiTaskModel('yolov8m', pretrained=True, small_box_assign=16).cuda().train()
    _, _, losses = forward(model, images, targets)
    assert all(loss.dtype == torch.float32 and loss.isfinite() and loss > 0 for loss in losses.values())
    assert 'select_candidates_in_gts' in vars(model.criterion.assigner)
    plain = YOLOMultiTaskModel('yolov8m', pretrained=False).cuda().train()
    forward(plain, images, targets)
    assert 'select_candidates_in_gts' not in vars(plain.criterion.assigner)


def test_neck_pooled_mlp_head_trains_the_neck_and_predict_rebuilds_it(tmp_path):
    model = YOLOMultiTaskModel('yolov8m', pretrained=False, cls_pool='neck', cls_head='mlp', cls_dropout=0.2).cuda()
    assert model.cls_head[0].normalized_shape == (192 + 384 + 576,)
    assert isinstance(model.cls_head[3], torch.nn.Dropout) and model.cls_head[-1].in_features == 512
    images, labels, targets = first_train_images(2)
    logits, _, _ = forward(model.train(), images, targets)
    F.cross_entropy(logits, labels).backward()
    p5_out = model.backbone.model[21]  # the neck's last C2f, after the SPPF: only reached through the neck maps
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in p5_out.parameters())
    torch.save(model.state_dict(), tmp_path / 'model.pt')
    (tmp_path / 'metrics.json').write_text(json.dumps({'args': {'backbone': 'yolov8m', 'adapt': 'full',
                                                                'cls_pool': 'neck', 'cls_head': 'mlp',
                                                                'cls_dropout': 0.2}}))
    model.eval()
    with torch.no_grad():
        expected, rebuilt = forward(model, images), forward(load_model(tmp_path), images)
    assert torch.allclose(expected[0], rebuilt[0])


def test_plain_head_keeps_the_layout_of_older_runs():
    assert list(YOLOMultiTaskModel('yolov8m', pretrained=False).cls_head.state_dict()) == [
        '0.weight', '0.bias', '1.weight', '1.bias']


@pytest.mark.parametrize('name', ['yolov8m', 'yolov8m-p2'])
def test_predict_rebuilds_the_model_from_a_run_folder(tmp_path, name):
    model = YOLOMultiTaskModel(name, pretrained=False).cuda().eval()
    torch.save(model.state_dict(), tmp_path / 'model.pt')
    (tmp_path / 'metrics.json').write_text(json.dumps({'args': {'backbone': name, 'adapt': 'full'}}))
    images = first_train_images(2)[0]
    with torch.no_grad():
        expected, rebuilt = forward(model, images), forward(load_model(tmp_path), images)
    assert torch.allclose(expected[0], rebuilt[0])
    assert all(torch.allclose(a['boxes'], b['boxes']) for a, b in zip(expected[1], rebuilt[1]))


@pytest.mark.parametrize('name', ['yolov8m', 'yolov8m-p2'])
def test_pretrained_yolov8m_overfits_16_training_images(name):
    torch.manual_seed(0)
    images, labels, targets = first_train_images(16)
    model = YOLOMultiTaskModel(name, pretrained=True).cuda().train()
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
