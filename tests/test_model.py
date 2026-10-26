"""sarbench/model.py on real images: outputs and losses, fp32 detection heads under bf16 autocast, one backbone pass
per task with MoE-LoRA, RetinaNet in place of Faster R-CNN, and overfit checks (a pretrained ViT fits 16 training
images, with either detector)."""
from pathlib import Path

import torch
import torch.nn.functional as F

from sarbench.adapters import MoELoRA, add_adapters
from sarbench.backbones import build_backbone
from sarbench.data import SARMultiTask, collate, to_original_xywh
from sarbench.metrics import detection_metrics
from sarbench.model import (RPN_MATCH, MultiTaskModel, SwinMultiTaskModel, anchor_similarity, gaussian_heatmaps,
                            heatmap_loss)

ROOT = Path(__file__).resolve().parents[1] / 'data'  # make_split.py's split, as train.slurm's --data data


def first_train_images(n):
    """The first n training images (no augmentation), their labels and their targets, on the GPU."""
    dataset = SARMultiTask(ROOT, 'train', limit=n)
    images, labels, targets = collate([dataset[i] for i in range(n)])
    return images.cuda(), labels.cuda(), [{**t, 'boxes': t['boxes'].cuda(), 'labels': t['labels'].cuda()}
                                          for t in targets]


def forward(model, images, targets=None):
    with torch.autocast('cuda', dtype=torch.bfloat16):
        return model(images, targets)


def test_training_forward_returns_logits_and_the_four_losses():
    model = MultiTaskModel(build_backbone('vit', pretrained=False), task_routing=False).cuda().train()
    images, _, targets = first_train_images(2)
    logits, detections, losses = forward(model, images, targets)
    assert logits.shape == (2, 9) and detections == []
    assert set(losses) == {'loss_objectness', 'loss_rpn_box_reg', 'loss_classifier', 'loss_box_reg'}
    assert all(loss.dtype == torch.float32 and loss.isfinite() for loss in losses.values())


def test_detection_heads_run_in_fp32_and_boxes_are_not_quantized():
    model = MultiTaskModel(build_backbone('vit', pretrained=False), task_routing=False).cuda().eval()
    dtypes = set()  # of the RPN's and the box head's outputs: bf16 here would snap boxes to a 2 px grid near 512
    model.rpn.head.register_forward_hook(lambda module, inputs, out: dtypes.update(t.dtype for t in out[0] + out[1]))
    model.roi_heads.box_predictor.register_forward_hook(lambda module, inputs, out: dtypes.update(t.dtype for t in out))
    with torch.no_grad():
        _, detections, losses = forward(model, first_train_images(2)[0])
    assert dtypes == {torch.float32} and losses == {}
    boxes, labels = torch.cat([d['boxes'] for d in detections]), torch.cat([d['labels'] for d in detections])
    assert len(boxes) > 0 and boxes.dtype == torch.float32 and detections[0]['scores'].dtype == torch.float32
    assert boxes.min() >= 0 and boxes.max() <= 512 and labels.min() >= 1 and labels.max() <= 9
    assert (boxes != boxes.bfloat16().float()).any()  # not every coordinate is on the bf16 grid


def test_moelora_runs_the_backbone_once_per_task():
    backbone = build_backbone('vit', pretrained=False)
    add_adapters(backbone, 'moelora')
    model = MultiTaskModel(backbone, task_routing=True).cuda().eval()
    moelora = next(m for m in model.modules() if isinstance(m, MoELoRA))
    tasks = []  # the routers' task at every backbone call
    model.backbone.register_forward_pre_hook(lambda module, inputs: tasks.append(moelora.task))
    with torch.no_grad():
        forward(model, first_train_images(2)[0])
    assert tasks == [0, 1]  # classification, then detection


def test_heatmap_targets_peak_at_box_centres_and_keep_close_objects_apart():
    # Two 8 px boxes centred on the stride-4 cells (row 10, col 20) and (row 12, col 20): 8 px apart, sigma 2 px
    boxes = torch.tensor([[78., 38., 86., 46.], [78., 46., 86., 54.]])
    heatmap, empty = gaussian_heatmaps([{'boxes': boxes}, {'boxes': torch.zeros(0, 4)}], 128, 4, min_sigma=2.0)
    assert heatmap.shape == (128, 128) and heatmap.max() == 1
    assert heatmap[10, 20] == 1 and heatmap[12, 20] == 1  # a peak at each centre
    assert heatmap[11, 20] < 0.5  # max, not sum: the cell between them stays low
    assert (empty == 0).all()


def test_heatmap_loss_is_lower_for_the_target_than_for_an_empty_map():
    target = gaussian_heatmaps([{'boxes': torch.tensor([[40., 40., 72., 72.]])}], 128, 4, min_sigma=2.0)
    assert heatmap_loss(torch.logit(target.clamp(1e-4, 1 - 1e-4)), target) < heatmap_loss(target * 0 - 10, target)


def test_heatmap_head_trains_only_and_predict_loads_without_it():
    model = MultiTaskModel(build_backbone('vit', pretrained=False), task_routing=False, heatmap=True).cuda().train()
    images, _, targets = first_train_images(2)
    _, _, losses = forward(model, images, targets)
    assert 'heatmap' in losses and losses['heatmap'].dtype == torch.float32 and losses['heatmap'].isfinite()
    with torch.no_grad():
        assert forward(model.eval(), images)[2] == {}  # the head is unused at inference
    plain = MultiTaskModel(build_backbone('vit', pretrained=False), task_routing=False)
    assert plain.heatmap_head is None
    plain.load_state_dict({k: v for k, v in model.state_dict().items() if not k.startswith('heatmap_head.')})


def test_retinanet_returns_its_two_losses_and_fp32_unquantized_boxes():
    model = MultiTaskModel(build_backbone('vit', pretrained=False), task_routing=False, detector='retinanet').cuda()
    assert not hasattr(model, 'rpn') and not hasattr(model, 'roi_heads')
    images, _, targets = first_train_images(2)
    _, detections, losses = forward(model.train(), images, targets)
    assert detections == [] and set(losses) == {'loss_focal', 'loss_retina_box_reg'}  # 'classification' stays free
    assert all(loss.dtype == torch.float32 and loss.isfinite() for loss in losses.values())
    model.retinanet.score_thresh = 0.0  # untrained, every score is near the 0.01 prior, below the 0.05 default
    with torch.no_grad():
        _, detections, losses = forward(model.eval(), images)
    assert losses == {} and len(detections) == 2
    boxes, labels = torch.cat([d['boxes'] for d in detections]), torch.cat([d['labels'] for d in detections])
    assert all(0 < len(d['boxes']) <= 100 for d in detections)
    assert boxes.dtype == torch.float32 and detections[0]['scores'].dtype == torch.float32
    assert boxes.min() >= 0 and boxes.max() <= 512 and labels.min() >= 1 and labels.max() <= 9  # as Faster R-CNN's
    assert (boxes != boxes.bfloat16().float()).any()  # not every coordinate is on the bf16 grid


def test_retinanet_overfits_16_training_images():
    """As the Faster R-CNN overfit check below, detection only: labels 1..9 must map to the focal-loss columns 0..8
    and back, or AP50 stays near 0."""
    torch.manual_seed(0)
    images, _, targets = first_train_images(16)
    model = MultiTaskModel(build_backbone('vit', pretrained=True), task_routing=False, detector='retinanet')
    model = model.cuda().train()
    heads = [p for name, p in model.named_parameters() if not name.startswith('backbone.')]
    optimizer = torch.optim.AdamW([{'params': model.backbone.parameters(), 'lr': 2e-5}, {'params': heads}], lr=1e-4)
    for _ in range(200):
        loss = sum(forward(model, images, targets)[2].values())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        detections = forward(model.eval(), images)[1]
    predictions = []
    for t, d in zip(targets, detections):
        boxes = to_original_xywh(d['boxes'], t['orig_size']).tolist()
        predictions += [{'image_id': t['image_id'], 'category_id': c, 'bbox': b, 'score': s}
                        for b, c, s in zip(boxes, d['labels'].tolist(), d['scores'].tolist())]
    image_ids = [t['image_id'] for t in targets]
    assert detection_metrics(ROOT / 'detection' / 'instances_train.json', predictions, image_ids)['AP50'] > 0.5


def test_nwd_keeps_shifted_tiny_boxes_and_max_keeps_shifted_large_ones():
    # An 8 px box with an 8 px anchor 2 px off in x and y; a 200 px box with a 200 px anchor 16 px off in x and y
    gt = torch.tensor([[100., 100., 108., 108.], [100., 100., 300., 300.]])
    anchors = torch.tensor([[102., 102., 110., 110.], [116., 116., 316., 316.]])
    iou, nwd, best = (anchor_similarity(gt, anchors, mode, c=16.0) for mode in ('iou', 'nwd', 'max'))
    assert iou[0, 0] < 0.4 < 0.7 < nwd[0, 0]  # tiny: not positive by IoU (0.39), positive by NWD (0.84)
    assert nwd[1, 1] < 0.3 < 0.7 < iou[1, 1]  # large: positive by IoU (0.73), background by NWD (0.24)
    assert torch.isclose(nwd[0, 0], torch.exp(torch.tensor(-8 ** 0.5 / 16)))  # W2 = |(2, 2, 0, 0)|
    assert torch.equal(best, torch.maximum(iou, nwd))


def test_rpn_match_sets_how_the_rpn_labels_anchors():
    backbone = torch.nn.Identity()
    backbone.embed_dim = 768  # all MultiTaskModel.__init__ reads from the backbone
    gt, anchors = torch.tensor([[100., 100., 108., 108.]]), torch.tensor([[102., 102., 110., 110.]])
    for mode in RPN_MATCH:
        rpn = MultiTaskModel(backbone, task_routing=False, rpn_match=mode, nwd_c=16.0).rpn
        assert torch.equal(rpn.box_similarity(gt, anchors), anchor_similarity(gt, anchors, mode, c=16.0))

def test_pretrained_vit_overfits_16_training_images():
    torch.manual_seed(0)
    images, labels, targets = first_train_images(16)
    model = MultiTaskModel(build_backbone('vit', pretrained=True), task_routing=False).cuda().train()
    heads = [p for name, p in model.named_parameters() if not name.startswith('backbone.')]
    optimizer = torch.optim.AdamW([{'params': model.backbone.parameters(), 'lr': 2e-5}, {'params': heads}], lr=1e-4)
    for _ in range(200):
        with torch.autocast('cuda', dtype=torch.bfloat16):
            logits, _, losses = model(images, targets)
            loss = F.cross_entropy(logits, labels) + sum(losses.values())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        logits, detections, _ = forward(model.eval(), images)
    assert (logits.argmax(dim=1) == labels).all()  # accuracy 100 % (16 images miss 2 classes: no macro metrics)
    predictions = []
    for t, d in zip(targets, detections):
        boxes = to_original_xywh(d['boxes'], t['orig_size']).tolist()
        predictions += [{'image_id': t['image_id'], 'category_id': c, 'bbox': b, 'score': s}
                        for b, c, s in zip(boxes, d['labels'].tolist(), d['scores'].tolist())]
    image_ids = [t['image_id'] for t in targets]
    assert detection_metrics(ROOT / 'detection' / 'instances_train.json', predictions, image_ids)['AP50'] > 0.9


def test_anchors_scale_with_the_image_size():
    sizes = lambda size: MultiTaskModel(build_backbone('vit', pretrained=False, img_size=size), task_routing=False,
                                        image_size=size).rpn.anchor_generator.sizes
    assert sizes(768) == tuple(tuple(1.5 * a for a in level) for level in sizes(512))


def test_swin_at_768_px_trains_and_detects_on_the_whole_image():
    images, _, targets = first_train_images(2)
    images = F.interpolate(images, size=768, mode='bilinear')
    targets = [{**t, 'boxes': t['boxes'] * 1.5} for t in targets]
    model = SwinMultiTaskModel(build_backbone('swin_t', pretrained=False, img_size=768), task_routing=False,
                               image_size=768).cuda()
    _, _, losses = forward(model.train(), images, targets)
    assert all(loss.isfinite() for loss in losses.values())
    with torch.no_grad():
        logits, detections, _ = forward(model.eval(), images)
    assert logits.shape == (2, 9) and len(detections) == 2
    assert all(d['boxes'].numel() == 0 or d['boxes'].max() <= 768 for d in detections)
