"""sarbench/model.py on real images: outputs and losses, fp32 detection heads under bf16 autocast, one backbone pass
per task with MoE-LoRA, and an overfit check (a pretrained ViT fits 16 training images)."""
from pathlib import Path

import torch
import torch.nn.functional as F

from sarbench.adapters import MoELoRA, add_adapters
from sarbench.backbones import build_backbone
from sarbench.data import SARMultiTask, collate, to_original_xywh
from sarbench.metrics import detection_metrics
from sarbench.model import MultiTaskModel, gaussian_heatmaps, heatmap_loss

ROOT = Path(__file__).resolve().parents[2] / 'dataset' / 'SARFact-Course-20K'


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
