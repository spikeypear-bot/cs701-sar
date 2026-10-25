"""The backbones, built for 512 px input (img_size: another size, a multiple of 32). The two ViT-B/16: images [B, 1, 512, 512] -> patch tokens [B, 1024, 768].
Swin and ConvNeXt: images [B, 1, 512, 512] -> 4 feature maps at strides 4, 8, 16 and 32. in_chans > 1: input channels
from SAR filters (data.input_channels); in_chans=3 keeps the RGB patch-embedding weights as they are."""
import timm
from terratorch.registry import BACKBONE_REGISTRY
from torch import nn


class ViT(nn.Module):
    """timm ViT-B/16 pretrained on ImageNet-21k."""

    def __init__(self, pretrained: bool, in_chans=1, img_size=512):
        super().__init__()
        # in_chans=1: timm sums the RGB patch-embedding weights; img_size: it resamples the position embedding
        self.vit = timm.create_model('vit_base_patch16_224.augreg_in21k', pretrained=pretrained,
                                     img_size=img_size, in_chans=in_chans, num_classes=0)
        self.embed_dim = 768

    @property
    def blocks(self):
        return self.vit.blocks

    @property
    def patch_embed(self):
        """The input layer (16 x 16 patches -> 768 numbers), the only one that sees the input channels."""
        return self.vit.patch_embed

    def forward(self, x):
        tokens = self.vit.forward_features(x)  # [B, 1 + N, 768], final norm applied
        return tokens[:, self.vit.num_prefix_tokens:]  # drop the CLS token


class TerraMind(nn.Module):
    """TerraMind-1.0-base (IBM/ESA) through its Sentinel-1 GRD input."""

    def __init__(self, pretrained: bool):
        super().__init__()
        self.terramind = BACKBONE_REGISTRY.build('terratorch_terramind_v1_base', pretrained=pretrained,
                                                 modalities=['S1GRD'])
        self.embed_dim = 768

    @property
    def blocks(self):
        return self.terramind.encoder

    def forward(self, x):
        per_block = self.terramind({'S1GRD': x.repeat(1, 2, 1, 1)})  # gray image as both VV and VH
        return per_block[-1]  # last block's tokens [B, N, 768]; terratorch already applied encoder_norm


# --backbone name -> timm model (architecture.pretrained weights); timm.list_pretrained('swin*') lists the others
SWIN_MODELS = {
    'swin_t': 'swin_tiny_patch4_window7_224.ms_in22k_ft_in1k',
    'swin_b': 'swin_base_patch4_window7_224.ms_in22k_ft_in1k',
    'swin_b384': 'swin_base_patch4_window12_384.ms_in22k_ft_in1k',  # same weights count, 384 px, 12 x 12 windows
    # Swin V2-B (cosine attention, log-spaced continuous position bias, post-norm): pretrained with 12 x 12 windows at
    # 192 px, fine-tuned with 16 x 16 at 256; 16 divides all four maps at 512, so no padding
    'swinv2_b': 'swinv2_base_window12to16_192to256.ms_in22k_ft_in1k',
}


class Swin(nn.Module):
    """A timm Swin Transformer (V1 or V2), its 4 stages as feature maps [B, C_i, 512 / s, 512 / s] for s = 4, 8, 16, 32.
    Pretrained at 224, 256 or 384 px, which is fine at 512: Swin's only position information is relative, inside each
    7 x 7 (12 x 12 for swin_b384, 16 x 16 for swinv2_b) window (the maps of 128, 64, 32 and 16 are padded to whole
    windows)."""

    def __init__(self, pretrained: bool, name: str, in_chans=1, img_size=512):
        super().__init__()
        self.swin = timm.create_model(name, pretrained=pretrained, features_only=True, out_indices=(0, 1, 2, 3),
                                      img_size=img_size, in_chans=in_chans)  # 1: timm sums the RGB patch-embedding weights
        for blk in self.blocks:
            if hasattr(blk.attn, 'qkv_bias_separate'):
                # V2 computes F.linear(x, self.qkv.weight, qkv_bias), which bypasses (and breaks on) a LoRA wrapper of
                # qkv; separate calls self.qkv(x) and then adds the bias (qkv itself has none), the same output
                blk.attn.qkv_bias_separate = True
        self.channels = self.swin.feature_info.channels()  # Swin-T [96, 192, 384, 768], Swin-B [128, ..., 1024]
        self.embed_dim = self.channels[-1]  # the last map's channels, the input of the classification head

    @property
    def blocks(self):
        """Every block of the 4 stages, for add_adapters (each has attn.qkv and attn.proj)."""
        return [blk for i in range(4) for blk in self.swin[f'layers_{i}'].blocks]

    @property
    def patch_embed(self):
        """The input layer (4 x 4 patches -> C numbers), the only one that sees the input channels."""
        return self.swin['patch_embed']

    def forward(self, x):
        return [m.permute(0, 3, 1, 2) for m in self.swin(x)]  # timm gives [B, H, W, C]; the FPN takes [B, C, H, W]


# --backbone name -> timm model. ImageNet-22k weights without the 1k fine-tuning, as the ConvNeXt paper's detection
# models (Liu et al., CVPR 2022, the official Cascade Mask R-CNN ConvNeXt-B config: convnext_base_22k_224.pth)
CONVNEXT_MODELS = {'convnext_b': 'convnext_base.fb_in22k'}


class ConvNeXt(nn.Module):
    """A timm ConvNeXt, its 4 stages as feature maps [B, C_i, 512 / s, 512 / s] for s = 4, 8, 16, 32, each through a
    new LayerNorm as in the paper's detection backbone (the official object_detection/backbone/convnext.py norm0-3).
    Fully convolutional: any multiple of 32 works as the input size. drop_path: stochastic depth rate, rising
    linearly over the blocks to this value (paper, ConvNeXt-B Cascade Mask R-CNN: 0.6)."""
    adapter_targets = ('mlp.fc1', 'mlp.fc2')  # the block's two 1 x 1 layers (nn.Linear on [B, H, W, C]); no attention

    def __init__(self, pretrained: bool, name: str, in_chans=1, drop_path=0.0):
        super().__init__()
        self.convnext = timm.create_model(name, pretrained=pretrained, features_only=True, out_indices=(0, 1, 2, 3),
                                          in_chans=in_chans, drop_path_rate=drop_path)  # 1: timm sums the RGB stem
        self.channels = self.convnext.feature_info.channels()  # ConvNeXt-B [128, 256, 512, 1024]
        self.embed_dim = self.channels[-1]
        # New, not pretrained: add_adapters keeps them trainable
        self.out_norms = nn.ModuleList(timm.layers.LayerNorm2d(c) for c in self.channels)
        # The convs get permuted views (the blocks' and LayerNorm2d's permute back from [B, H, W, C]), on which cuDNN
        # (torch 2.9.0, bf16 autocast) failed with CUDNN_STATUS_BAD_PARAM in the depthwise conv (job 338636); a plain
        # contiguous [B, C, H, W] copy is the layout every other conv here gets
        for conv in self.convnext.modules():
            if isinstance(conv, nn.Conv2d):
                conv.register_forward_pre_hook(lambda module, args: (args[0].contiguous(),))

    @property
    def blocks(self):
        """Every block of the 4 stages, for add_adapters and --unfreeze-blocks (ConvNeXt-B: 0-2 stage 1, 3-5 stage 2,
        6-32 stage 3, 33-35 stage 4)."""
        return [blk for i in range(4) for blk in self.convnext[f'stages_{i}'].blocks]

    @property
    def patch_embed(self):
        """The input layer (4 x 4 conv + LayerNorm), the only one that sees the input channels."""
        return nn.ModuleList([self.convnext['stem_0'], self.convnext['stem_1']])

    def forward(self, x):
        return [norm(m) for norm, m in zip(self.out_norms, self.convnext(x))]  # already [B, C, H, W]


def build_backbone(name, pretrained, in_chans=1, img_size=512, drop_path=0.0):
    """name: 'vit', 'terramind' or a key of SWIN_MODELS or CONVNEXT_MODELS. pretrained=False builds the same
    architecture with random weights. in_chans: input channels (ViT, Swin and ConvNeXt; TerraMind takes the gray image
    only). img_size: input side in pixels (ViT and Swin; ConvNeXt takes any; TerraMind takes 512 only). drop_path:
    ConvNeXt only, see ConvNeXt."""
    if name in CONVNEXT_MODELS:
        return ConvNeXt(pretrained, CONVNEXT_MODELS[name], in_chans, drop_path)
    if name in SWIN_MODELS:
        return Swin(pretrained, SWIN_MODELS[name], in_chans, img_size)
    if name == 'terramind':
        if in_chans != 1 or img_size != 512:
            raise ValueError('TerraMind takes 1 input channel at 512 px')
        return TerraMind(pretrained)
    return ViT(pretrained, in_chans, img_size)
