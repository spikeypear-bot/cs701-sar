"""SimMIM (Xie et al., 2022, arXiv 2111.09886; code github.com/microsoft/SimMIM) as parameter-efficient extended
pretraining of a Swin backbone on the course's own training images, after ExPLoRA (Khanna et al., 2025, arXiv
2406.10973): the ImageNet weights stay frozen, LoRA adapters on every block (and any unfrozen blocks) learn to
reconstruct hidden parts of SAR images. Fine-tuning (train.py --ssl-weights) puts the saved adapters back, frozen and
unmerged, under new task adapters, so the Swin weights stay exactly the published ones and what SSL trained can be
counted.

Changes from SimMIM: LoRA instead of training every weight; 512 px with the backbone's own windows (SimMIM pretrains
Swin at 192 px with 6 x 6 windows); the final LayerNorm, which timm's features_only model leaves out, belongs to the
decoder here."""
import math

import torch
import torch.nn.functional as F
from torch import nn

from .adapters import add_adapters, unfreeze_blocks


class SimMIM(nn.Module):
    """A Swin backbone (backbones.Swin) with SimMIM's masking and decoder: in every image a mask_ratio share of the
    block x block squares is hidden, their patch tokens replaced by one learned token after the patch embedding; a
    LayerNorm and a 1 x 1 conv on the stride-32 map predict the block x block pixels under each position
    (PixelShuffle), and the loss is L1 on the hidden pixels only. The mask token, norm and decoder are for
    pretraining only (Swin-B: 1.05 M parameters)."""

    def __init__(self, backbone, mask_ratio=0.6, block=32):
        super().__init__()
        self.backbone = backbone
        self.mask_ratio, self.block = mask_ratio, block
        self.in_chans = backbone.patch_embed.proj.in_channels
        self.mask_token = nn.Parameter(torch.zeros(backbone.channels[0]))
        nn.init.trunc_normal_(self.mask_token, std=0.02)  # as SimMIM
        self.norm = nn.LayerNorm(backbone.embed_dim)
        stride = 32  # of the last Swin map
        self.decoder = nn.Sequential(nn.Conv2d(backbone.embed_dim, stride ** 2 * self.in_chans, 1),
                                     nn.PixelShuffle(stride))

    def random_masks(self, n, height, width, generator=None):
        """[n, height / block, width / block] of 0 (seen) and 1 (hidden), the same number of hidden squares in
        every image (rounded up, as SimMIM's MaskGenerator)."""
        rows, cols = height // self.block, width // self.block
        hidden = math.ceil(self.mask_ratio * rows * cols)
        device = generator.device if generator is not None else self.mask_token.device
        order = torch.rand(n, rows * cols, generator=generator, device=device).argsort(dim=1)
        masks = torch.zeros(n, rows * cols, device=device).scatter_(1, order[:, :hidden], 1.0)
        return masks.view(n, rows, cols)

    def forward(self, images, masks):
        """images [B, C, H, W], masks [B, H / block, W / block] (1 = hidden) -> the L1 loss on the hidden pixels."""
        swin = self.backbone.swin
        x = swin['patch_embed'](images)  # [B, H / 4, W / 4, C0], timm's NHWC
        per_token = self.block // (images.shape[-1] // x.shape[2])  # tokens per square side: 32 / 4 = 8
        hidden = masks.repeat_interleave(per_token, 1).repeat_interleave(per_token, 2).unsqueeze(-1).to(x.dtype)
        x = x * (1 - hidden) + self.mask_token.to(x.dtype) * hidden
        for i in range(4):
            x = swin[f'layers_{i}'](x)
        reconstruction = self.decoder(self.norm(x).permute(0, 3, 1, 2)).float()  # [B, C, H, W]
        hidden = masks.repeat_interleave(self.block, 1).repeat_interleave(self.block, 2).unsqueeze(1)
        error = F.l1_loss(reconstruction, images.float(), reduction='none')
        return (error * hidden).sum() / (hidden.sum() * self.in_chans + 1e-5)


def add_ssl_adapters(backbone, rank, unfreeze=()):
    """LoRA of `rank` (alpha 2 x rank) on attn.qkv and attn.proj of every block, then backbone.blocks[i] fully
    trainable for i in `unfreeze`; everything else frozen. Returns the pretrained weights made trainable, so they can
    get their own learning rate (the adapters in those blocks are not among them)."""
    add_adapters(backbone, 'lora', rank)
    return unfreeze_blocks(backbone, unfreeze)


def ssl_checkpoint(model, name, rank, unfreeze):
    """What fine-tuning needs from a SimMIM model: the backbone weights SSL trained (adapters and unfrozen blocks),
    and how many parameters SSL trained in all, the pretraining-only ones included, for the trainable limit."""
    state = {n: p.detach().cpu() for n, p in model.backbone.named_parameters() if p.requires_grad}
    trained = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'backbone': name, 'rank': rank, 'unfreeze_blocks': list(unfreeze), 'state': state,
            'trained_parameters': trained, 'pretraining_only_parameters': trained - sum(t.numel() for t in state.values())}


def load_ssl(backbone, checkpoint):
    """Put the adapters of an ssl_checkpoint on a newly built backbone with their weights and those of the blocks
    SSL unfroze, all frozen; task adapters (add_adapters) then wrap the adapted layers. Returns the parameters SSL
    trained, so train.py can count them with its own."""
    add_adapters(backbone, 'lora', checkpoint['rank'])  # also freezes the backbone
    _, unexpected = backbone.load_state_dict(checkpoint['state'], strict=False)  # missing: the frozen ImageNet ones
    if unexpected:
        raise ValueError(f'SSL weights that the backbone does not have: {unexpected[:3]} ...')
    parameters = dict(backbone.named_parameters())
    return [parameters[n] for n in checkpoint['state']]
