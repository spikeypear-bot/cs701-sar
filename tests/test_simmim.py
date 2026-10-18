"""sarbench/simmim.py: masks, a loss on hidden pixels only that hidden pixels cannot leak into, which weights SSL
trains, and the trip from an SSL checkpoint through train.py's construction to predict.py's load_model."""
import json

import torch
from torch import nn

from predict import load_model
from sarbench.adapters import LoRA, MoELoRA, add_adapters
from sarbench.backbones import build_backbone
from sarbench.model import SwinMultiTaskModel
from sarbench.simmim import SimMIM, add_ssl_adapters, load_ssl, ssl_checkpoint

LORA_WEIGHTS = ('.down.weight', '.up.weight')


class Fixed(nn.Module):
    """A decoder that ignores its input and returns `out`."""

    def __init__(self, out):
        super().__init__()
        self.out = out

    def forward(self, x):
        return self.out


def ssl_model(rank=8, unfreeze=(11,)):
    """SimMIM on a random-init Swin-T (seed 0) whose SSL adapters have a non-zero B and whose unfrozen blocks are
    changed, as if trained, so both change the output."""
    torch.manual_seed(0)
    backbone = build_backbone('swin_t', pretrained=False)
    with torch.no_grad():
        for p in add_ssl_adapters(backbone, rank, unfreeze):
            p.add_(0.01 * torch.randn_like(p))
    for layer in backbone.modules():
        if isinstance(layer, LoRA):
            nn.init.normal_(layer.up.weight, std=0.02)
    return SimMIM(backbone).cuda().eval()


def test_every_image_hides_the_same_number_of_squares():
    masks = ssl_model().random_masks(3, 512, 512)
    assert masks.shape == (3, 16, 16)
    assert (masks.sum(dim=(1, 2)) == 154).all()  # ceil(0.6 x 256)
    assert not torch.equal(masks[0], masks[1])


def test_loss_is_l1_on_hidden_pixels_only():
    model = ssl_model()
    images = torch.randn(2, 1, 512, 512, device='cuda')
    prediction = torch.randn_like(images)
    model.decoder = Fixed(prediction)
    masks = model.random_masks(2, 512, 512)
    hidden = masks.repeat_interleave(32, 1).repeat_interleave(32, 2).unsqueeze(1).bool()
    with torch.no_grad():
        loss = model(images, masks)
    torch.testing.assert_close(loss, (prediction - images).abs()[hidden].mean())


def test_hidden_pixels_do_not_reach_the_encoder():
    model = ssl_model()
    images = torch.randn(1, 1, 512, 512, device='cuda')
    masks = model.random_masks(1, 512, 512)
    hidden = masks.repeat_interleave(32, 1).repeat_interleave(32, 2).unsqueeze(1).bool()
    changed = torch.where(hidden, torch.randn_like(images), images)
    outputs = []
    model.decoder.register_forward_hook(lambda module, args, out: outputs.append(out))
    with torch.no_grad():
        model(images, masks), model(changed, masks)
    torch.testing.assert_close(outputs[0], outputs[1])


def test_ssl_trains_adapters_unfrozen_blocks_and_its_own_modules():
    model = ssl_model(unfreeze=(11,))
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    block = {f'backbone.{n}' for n, _ in model.backbone.swin.named_parameters(prefix='swin') if '.layers_3.blocks.1.' in n}
    adapters = {n for n, _ in model.named_parameters() if n.endswith(LORA_WEIGHTS)}
    own = {n for n, _ in model.named_parameters() if not n.startswith('backbone.')}
    assert own == {'mask_token', 'norm.weight', 'norm.bias', 'decoder.0.weight', 'decoder.0.bias'}
    assert trainable == adapters | block | own


def test_fine_tuning_starts_from_the_ssl_backbone():
    """load_ssl, then MoE-LoRA (B = 0) on top, gives the SSL backbone's maps; only MoE-LoRA trains."""
    model = ssl_model()
    checkpoint = ssl_checkpoint(model, 'swin_t', 8, [11])
    torch.manual_seed(0)  # the same random "pretrained" weights as ssl_model
    backbone = build_backbone('swin_t', pretrained=False)
    ssl_trained = load_ssl(backbone, checkpoint)
    add_adapters(backbone, 'moelora')
    assert sum(p.numel() for p in ssl_trained) == sum(t.numel() for t in checkpoint['state'].values())
    assert all(isinstance(m.base, LoRA) for m in backbone.modules() if isinstance(m, MoELoRA))
    assert not any(p.requires_grad for p in ssl_trained)
    x = torch.randn(2, 1, 512, 512, device='cuda')
    with torch.no_grad():
        expected, actual = model.backbone(x), backbone.cuda().eval()(x)
    for e, a in zip(expected, actual):
        torch.testing.assert_close(a, e)


def test_predict_rebuilds_a_run_with_ssl_weights(tmp_path):
    checkpoint = ssl_checkpoint(ssl_model(), 'swin_t', 8, [11])
    backbone = build_backbone('swin_t', pretrained=False)
    load_ssl(backbone, checkpoint)
    add_adapters(backbone, 'moelora')
    model = SwinMultiTaskModel(backbone, task_routing=True)
    torch.save(model.state_dict(), tmp_path / 'model.pt')
    args = {'backbone': 'swin_t', 'adapt': 'moelora', 'input_channels': 'raw', 'ssl_weights': 'ssl.pt', 'ssl_rank': 8}
    (tmp_path / 'metrics.json').write_text(json.dumps({'args': args}))
    loaded = load_model(tmp_path)  # load_state_dict is strict: every key must match
    assert all(torch.equal(loaded.state_dict()[k].cpu(), v) for k, v in model.state_dict().items())
