"""The multi-task model on an Ultralytics YOLO network (YOLOv8 or YOLO11, COCO-pretrained): the image class from the
end of its backbone, the detections from its own neck and anchor-free head, in place of the backbone + FPN + Faster
R-CNN of model.py. Same interface as MultiTaskModel, so train.py, predict.py and the metrics stay as they are.
ultralytics is imported only here, and only when a model is built: the other runs do not need it."""
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
from torchvision.ops import box_convert, clip_boxes_to_image

YOLO_MODELS = ('yolov8s', 'yolov8m', 'yolov8l', 'yolo11s', 'yolo11m')  # the choices of --backbone; parameters with
# COCO's 80 classes (a little fewer at 9): v8s 11.2 M, v8m 25.9 M, v8l 43.7 M, 11s 9.4 M, 11m 20.1 M. v8l is over the
# 40 M trainable limit unless its first 6 layers are frozen (freeze_layers=6: 39.7 M trainable at 9 classes)
WEIGHTS_URL = 'https://github.com/ultralytics/assets/releases/download/v8.3.0/{}.pt'  # COCO detection weights


def pretrained_state(name, in_chans):
    """The COCO weights of `name` as a state dict, downloaded once into the torch hub cache. The first conv sees
    in_chans channels instead of RGB: its weights are summed over RGB and shared out over the new channels, as timm
    does for the other backbones (in_chans=1: the plain sum)."""
    from ultralytics.nn.tasks import torch_safe_load
    path = Path(torch.hub.get_dir()) / 'ultralytics' / f'{name}.pt'
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.hub.download_url_to_file(WEIGHTS_URL.format(name), str(path))
    checkpoint, _ = torch_safe_load(str(path))  # a pickled DetectionModel, in fp16
    state = checkpoint['model'].float().state_dict()
    if in_chans != 3:
        first = state['model.0.conv.weight']  # [out, 3, k, k]
        state['model.0.conv.weight'] = first.sum(dim=1, keepdim=True).repeat(1, in_chans, 1, 1) / in_chans
    return state


class YOLOMultiTaskModel(nn.Module):
    """images [B, C, H, W] -> classification logits (global average of the SPPF map -> LayerNorm -> Linear) and
    YOLO detections. `backbone` is the whole YOLO network (backbone, neck and head): it is pretrained, so train.py gives
    it --backbone-lr, and the new classification head --lr. Only the head's last class convs start untrained (9
    classes instead of COCO's 80). freeze_layers=N freezes YOLO layers 0..N-1, as Ultralytics' freeze=N (their BatchNorm
    statistics still adapt to SAR in training mode; they are buffers, not trained parameters).
    Training: YOLO's own loss (task-aligned assignment, BCE classes, CIoU + DFL boxes; Ultralytics' gains 7.5, 0.5,
    1.5), per image. Inference: Ultralytics' validation settings, confidence >= 0.001, NMS at IoU 0.7, one box may
    carry several classes; at most 100 detections per image, as COCO mAP counts."""

    def __init__(self, name, pretrained: bool, in_chans=1, num_classes=9, freeze_layers=0, score_thresh=0.001,
                 nms_thresh=0.7, detections_per_img=100):
        super().__init__()
        from ultralytics.nn.tasks import DetectionModel
        self.backbone = DetectionModel(f'{name}.yaml', ch=in_chans, nc=num_classes, verbose=False)
        self.backbone.args = SimpleNamespace(box=7.5, cls=0.5, dfl=1.5)  # the loss gains v8DetectionLoss reads
        self.transferred = 0  # pretrained tensors loaded, for the log
        if pretrained:
            own = self.backbone.state_dict()
            state = {k: v for k, v in pretrained_state(name, in_chans).items() if k in own and v.shape == own[k].shape}
            self.backbone.load_state_dict(state, strict=False)
            self.transferred = len(state)
        layers = self.backbone.model
        layers[:freeze_layers].requires_grad_(False)
        # SPPF ends YOLOv8's backbone; YOLO11 has one attention block (C2PSA) after it
        self.sppf = next(m.i for m in layers if type(m).__name__ == 'SPPF')
        channels = layers[self.sppf].cv2.conv.out_channels  # v8m: 576
        self.cls_head = nn.Sequential(nn.LayerNorm(channels), nn.Linear(channels, num_classes))
        self.score_thresh, self.nms_thresh, self.detections_per_img = score_thresh, nms_thresh, detections_per_img
        self.criterion = None  # v8DetectionLoss, made on the first training step: it takes the model's device then

    def features(self, x):
        """Every layer but the detection head, as Ultralytics' _predict_once -> the outputs of all layers."""
        outputs = []
        for m in self.backbone.model[:-1]:
            if m.f != -1:  # input from earlier layers (Concat), not only the previous one
                x = outputs[m.f] if isinstance(m.f, int) else [x if j == -1 else outputs[j] for j in m.f]
            x = m(x)
            outputs.append(x)
        return outputs

    def yolo_losses(self, raw, targets, images):
        """Ultralytics' loss on the head's raw maps; their targets are class 0..8 and normalized centre-xywh boxes.
        Their loss is summed over the batch, so it is divided by the batch size here, as the other detectors' losses
        are per image."""
        if self.criterion is None:
            from ultralytics.utils.loss import v8DetectionLoss
            self.criterion = v8DetectionLoss(self.backbone)
        h, w = images.shape[-2:]
        batch = {'batch_idx': torch.cat([torch.full((len(t['boxes']),), i, device=images.device, dtype=torch.float32)
                                         for i, t in enumerate(targets)]),
                 'cls': torch.cat([t['labels'] - 1 for t in targets]).float(),
                 'bboxes': torch.cat([box_convert(t['boxes'], 'xyxy', 'cxcywh') for t in targets])
                           / torch.tensor([w, h, w, h], device=images.device, dtype=torch.float32)}
        loss, _ = self.criterion(raw, batch)  # [box, cls, dfl], times the batch size
        loss = loss / len(images)
        return {'loss_yolo_box': loss[0], 'loss_yolo_cls': loss[1], 'loss_yolo_dfl': loss[2]}

    def postprocess(self, raw, size):
        """The head's raw maps -> per-image {'boxes' XYXY on the input, 'labels' 1..9, 'scores'}, as torchvision's."""
        from ultralytics.utils.ops import non_max_suppression
        predictions = self.backbone.model[-1]._inference(raw)  # [B, 4 + classes, anchors]: centre-xywh px, scores
        # max_time_img: Ultralytics stops NMS after 0.05 s per image and leaves the remaining images empty, which an
        # early, unsure model can hit at confidence 0.001; here no time limit
        kept = non_max_suppression(predictions, self.score_thresh, self.nms_thresh, multi_label=True,
                                   max_det=self.detections_per_img, max_time_img=1e6)  # [n, 6]: xyxy score class
        return [{'boxes': clip_boxes_to_image(d[:, :4], size), 'scores': d[:, 4], 'labels': d[:, 5].long() + 1}
                for d in kept]

    def forward(self, images, targets=None):
        """As MultiTaskModel.forward: images, targets (training only) -> (logits [B, num_classes], per-image detections
        (eval mode), {'loss_yolo_box', 'loss_yolo_cls', 'loss_yolo_dfl'} (training mode)).
        Precision rule as model.py's: the head's convs run in the caller's bf16, as RetinaNet's, and its raw maps go to
        fp32 before boxes are decoded, so no coordinate snaps to bf16's 2 px grid near 512."""
        outputs = self.features(images)
        logits = self.cls_head(outputs[self.sppf].mean(dim=(2, 3)))
        head = self.backbone.model[-1]  # Detect, on the P3, P4, P5 maps of the neck (strides 8, 16, 32)
        raw = [torch.cat((head.cv2[i](outputs[j]), head.cv3[i](outputs[j])), 1) for i, j in enumerate(head.f)]
        with torch.autocast(images.device.type, enabled=False):
            raw = [r.float() for r in raw]
            if self.training:
                return logits, [], self.yolo_losses(raw, targets, images)
            return logits, self.postprocess(raw, tuple(images.shape[-2:])), {}
