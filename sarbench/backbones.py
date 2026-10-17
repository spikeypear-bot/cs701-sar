"""The backbones, built for 512 px input. The two ViT-B/16: images [B, 1, 512, 512] -> patch tokens [B, 1024, 768].
Swin: images [B, 1, 512, 512] -> 4 feature maps at strides 4, 8, 16 and 32. in_chans > 1 (ViT, Swin): input channels
from SAR filters (data.input_channels); in_chans=3 keeps the RGB patch-embedding weights as they are."""
import timm
from terratorch.registry import BACKBONE_REGISTRY
from torch import nn


class ViT(nn.Module):
    """timm ViT-B/16 pretrained on ImageNet-21k."""

    def __init__(self, pretrained: bool, in_chans=1):
        super().__init__()
        # in_chans=1: timm sums the RGB patch-embedding weights; img_size=512: it resamples the position embedding
        self.vit = timm.create_model('vit_base_patch16_224.augreg_in21k', pretrained=pretrained,
                                     img_size=512, in_chans=in_chans, num_classes=0)
        self.embed_dim = 768

    @property
    def blocks(self):
        return self.vit.blocks

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
}


class Swin(nn.Module):
    """A timm Swin Transformer, its 4 stages as feature maps [B, C_i, 512 / s, 512 / s] for s = 4, 8, 16, 32.
    Pretrained at 224 px, which is fine at 512: Swin's only position information is relative, inside each 7 x 7
    window (the maps of 128, 64, 32 and 16 are padded to whole windows)."""

    def __init__(self, pretrained: bool, name: str, in_chans=1):
        super().__init__()
        self.swin = timm.create_model(name, pretrained=pretrained, features_only=True, out_indices=(0, 1, 2, 3),
                                      img_size=512, in_chans=in_chans)  # 1: timm sums the RGB patch-embedding weights
        self.channels = self.swin.feature_info.channels()  # Swin-T [96, 192, 384, 768], Swin-B [128, ..., 1024]
        self.embed_dim = self.channels[-1]  # the last map's channels, the input of the classification head

    @property
    def blocks(self):
        """Every block of the 4 stages, for add_adapters (each has attn.qkv and attn.proj)."""
        return [blk for i in range(4) for blk in self.swin[f'layers_{i}'].blocks]

    def forward(self, x):
        return [m.permute(0, 3, 1, 2) for m in self.swin(x)]  # timm gives [B, H, W, C]; the FPN takes [B, C, H, W]


def build_backbone(name, pretrained, in_chans=1):
    """name: 'vit', 'terramind' or a key of SWIN_MODELS. pretrained=False builds the same architecture with random
    weights. in_chans: input channels (ViT and Swin; TerraMind takes the gray image only)."""
    if name in SWIN_MODELS:
        return Swin(pretrained, SWIN_MODELS[name], in_chans)
    if name == 'terramind':
        if in_chans != 1:
            raise ValueError('TerraMind takes 1 input channel')
        return TerraMind(pretrained)
    return ViT(pretrained, in_chans)
