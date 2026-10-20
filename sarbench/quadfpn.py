"""Quad-FPN (Zhang, Zhang and Ke, Remote Sensing 2021, 13(14), 2771; SAR ship detection) in YOLO's neck. The paper
chains four FPNs after a ResNet-50 (five levels, 256 channels, Faster R-CNN in MMDetection):
- de: DE-CO-FPN, deformable 3 x 3 convs on the backbone maps that enter the pyramid;
- ca: CA-FR-FPN, the top-down path upsampled by CARAFE (Wang et al. 2019): a 1 x 1 conv amplifies the features 8 x,
  a content encoder predicts a 5 x 5 kernel per output pixel (pixel shuffle, softmax), and each output pixel is that
  kernel's weighted sum of its source pixel's 5 x 5 neighbours; in place of nearest-neighbour upsampling;
- pa: PA-SA-FPN, a bottom-up path (PANet) whose maps go through a space attention before each down-sampling:
  channel mean and max -> space encoder -> sigmoid -> times the map;
- bs: BS-GA-FPN, Libra R-CNN's balanced pyramid: every level resized to the middle one (max pooling or nearest
  upsampling), averaged, refined by a non-local block (embedded Gaussian, Wang et al. 2018), resized back.

YOLOv8/11's neck is already a PANet (top-down, then bottom-up, C2f/C3k2 fusions), so the four parts go into it rather
than replacing it, and its COCO weights are kept:
- de on the backbone maps as the neck reads them (v8: layers 4, 6, 9; 11: 4, 6, 10): the backbone itself and the
  image-class head see the plain maps; ca in place of the neck's two nn.Upsample, pa on the inputs of its two
  stride-2 convs, bs on the three maps the Detect head reads;
- every part is a residual branch behind a gate that starts at 0 (a BatchNorm scale per channel for de, a scalar for
  the others), so step 0 is the pretrained YOLO. Not zero-initialised convs: under Adam every weight moves by about
  the learning rate per step whatever its gradient, so a zero-initialised conv with n inputs can grow its output by
  about n x lr x the input per step (1e-3 x 256 inputs: a quarter of the input per step), the likely cause of the
  first run's collapse (yolov8m, all parts: val mAP 0 from epoch 4 on); a scalar gate grows by at most lr per step;
- de's offsets are its conv's output x 0.1 (as MMDetection's 0.1 learning rate for offsets): unscaled, they too grow
  by n x lr pixels per step;
- the non-local block scales its similarities by 1 / sqrt(channels), as MMCV's default (Libra R-CNN turns it off);
- CARAFE amplifies to a fixed 2048 channels (the paper's 8 x 256) rather than 8 x the map's 384-576: 2-5 M parameters
  less per upsampler, which keeps yolov8m under 40 M trainable;
- the paper does not give its space encoder: here 7 x 7 conv to 16 channels, ReLU, 7 x 7 conv to 1;
- YOLO's three output levels differ in width, so bs projects them to 256 channels and back (1 x 1 convs).
New parameters at 1 channel: yolov8m 11.35 M (de 4.83, ca 5.66, pa 0.003, bs 0.85) -> 37.2 M; yolo11m 14.04 M -> 34.1 M;
yolo11l -> 39.3 M; yolov8l is over the limit even with --freeze-layers 6."""
import torch
import torch.nn.functional as F
from torch import nn
from torchvision.ops import DeformConv2d

QUAD_FPN_PARTS = ('de', 'ca', 'pa', 'bs')


def width(layer):
    """Output channels of a YOLO backbone or neck block: C2f, C3k2, SPPF and C2PSA all end on their cv2 Conv."""
    return layer.cv2.conv.out_channels


class DeformableConv(nn.Module):
    """DE-CO: x + SiLU(BN(deformable 3 x 3 conv(x))), YOLO's Conv block with a deformable conv. Offsets: 0.1 x a
    3 x 3 conv, zero at the start (a plain 3 x 3 conv); the BN scale starts at 0, so the block starts as the
    identity."""

    def __init__(self, channels, kernel=3, offset_scale=0.1):
        super().__init__()
        self.offset_scale = offset_scale
        self.offset = nn.Conv2d(channels, 2 * kernel * kernel, kernel, padding=kernel // 2)
        nn.init.zeros_(self.offset.weight)
        nn.init.zeros_(self.offset.bias)
        self.conv = DeformConv2d(channels, channels, kernel, padding=kernel // 2, bias=False)
        self.bn = nn.BatchNorm2d(channels, eps=1e-3, momentum=0.03)  # as Ultralytics' BN
        nn.init.zeros_(self.bn.weight)

    def forward(self, x):
        # under autocast, torchvision runs deform_conv2d in fp32 and returns x's dtype
        return x + F.silu(self.bn(self.conv(x, self.offset_scale * self.offset(x))))


class CARAFE(nn.Module):
    """CA-FR: 2 x upsampling, nearest + gate x (CARAFE - nearest), the gate 0 at the start. CARAFE: 1 x 1 conv to
    `amplified` channels -> 3 x 3 content encoder to 4 x 25 -> pixel shuffle -> softmax: a 5 x 5 kernel per output
    pixel [B, 25, 2H, 2W], applied to the 5 x 5 neighbourhood of its source pixel."""

    def __init__(self, channels, amplified=2048, kernel=5, encoder_kernel=3, scale=2):
        super().__init__()
        self.kernel, self.scale = kernel, scale
        self.amplify = nn.Conv2d(channels, amplified, 1)
        self.encode = nn.Conv2d(amplified, scale ** 2 * kernel ** 2, encoder_kernel, padding=encoder_kernel // 2)
        self.gate = nn.Parameter(torch.zeros(()))

    def forward(self, x):
        b, c, h, w = x.shape
        k, s = self.kernel, self.scale
        nearest = F.interpolate(x, scale_factor=s, mode='nearest')
        kernels = F.pixel_shuffle(self.encode(self.amplify(x)), s).softmax(dim=1)  # [B, k * k, sH, sW]
        kernels = kernels.view(b, k * k, h, s, w, s).to(x.dtype)  # output pixel (h s + i, w s + j) -> [.., h, i, w, j]
        patches = F.unfold(x, k, padding=k // 2).view(b, c, k * k, h, w)  # each source pixel's neighbourhood
        out = torch.einsum('bckhw,bkhiwj->bchiwj', patches, kernels).reshape(b, c, s * h, s * w)
        return nearest + self.gate.to(x.dtype) * (out - nearest)  # in x's dtype: the neck's Concats stay bf16


class SpaceAttention(nn.Module):
    """PA-SA: x times 1 + gate (2 sigmoid(space encoder([channel mean, channel max])) - 1), a weight per pixel; at
    gate 1, the paper's sigmoid weights (times 2); the gate starts at 0, so the weights start at 1."""

    def __init__(self, hidden=16, kernel=7):
        super().__init__()
        self.encode = nn.Sequential(nn.Conv2d(2, hidden, kernel, padding=kernel // 2), nn.ReLU(),
                                    nn.Conv2d(hidden, 1, kernel, padding=kernel // 2))
        self.gate = nn.Parameter(torch.zeros(()))

    def forward(self, x):
        pooled = torch.cat([x.mean(dim=1, keepdim=True), x.amax(dim=1, keepdim=True)], dim=1)
        return x * (1 + self.gate.to(x.dtype) * (2 * torch.sigmoid(self.encode(pooled)) - 1))


class NonLocal(nn.Module):
    """GA refinement: x + gate W_O softmax(theta(x)^T phi(x) / sqrt(channels)) g(x) over all positions, 1 x 1 convs,
    no channel reduction (as Libra R-CNN's NonLocal2d); the gate starts at 0."""

    def __init__(self, channels):
        super().__init__()
        self.theta, self.phi, self.g = (nn.Conv2d(channels, channels, 1) for _ in range(3))
        self.out = nn.Conv2d(channels, channels, 1)
        self.gate = nn.Parameter(torch.zeros(()))

    def forward(self, x):
        b, c, h, w = x.shape
        q, k, v = (conv(x).flatten(2).transpose(1, 2) for conv in (self.theta, self.phi, self.g))  # [B, HW, C]
        o = F.scaled_dot_product_attention(q, k, v)  # scale 1 / sqrt(channels)
        return x + self.gate.to(x.dtype) * self.out(o.transpose(1, 2).reshape(b, c, h, w))


class BalancedPyramid(nn.Module):
    """BS-GA: the levels (largest first) -> 1 x 1 convs to `hidden` channels -> resized to the middle level (max pooling
    the larger ones, nearest upsampling the smaller) -> mean -> non-local block -> resized back to each level -> 1 x 1
    convs to its channels -> times the level's gate, which starts at 0 -> added to the level."""

    def __init__(self, channels, hidden=256):
        super().__init__()
        self.into = nn.ModuleList(nn.Conv2d(c, hidden, 1) for c in channels)
        self.refine = NonLocal(hidden)
        self.back = nn.ModuleList(nn.Conv2d(hidden, c, 1) for c in channels)
        self.gates = nn.Parameter(torch.zeros(len(channels)))

    @staticmethod
    def resize(x, size):
        return F.adaptive_max_pool2d(x, size) if x.shape[-1] > size[-1] else F.interpolate(x, size=size, mode='nearest')

    def forward(self, maps):
        size = maps[len(maps) // 2].shape[-2:]
        balanced = torch.stack([self.resize(conv(x), size) for conv, x in zip(self.into, maps)]).mean(dim=0)
        balanced = self.refine(balanced)
        return [x + gate.to(x.dtype) * conv(self.resize(balanced, x.shape[-2:]))
                for conv, gate, x in zip(self.back, self.gates, maps)]


class QuadFPN(nn.Module):
    """The chosen `parts` of Quad-FPN (QUAD_FPN_PARTS) for the Ultralytics YOLOv8/11 network `layers`, placed by its
    structure: the neck runs from the first nn.Upsample to the Detect head; its Concats read backbone maps (and the
    last backbone layer feeds the first Upsample); its stride-2 Convs after the last Upsample make the bottom-up
    path."""

    def __init__(self, layers, parts):
        super().__init__()
        upsamples = [m.i for m in layers if isinstance(m, nn.Upsample)]
        neck, head = layers[upsamples[0]:-1], layers[-1]
        laterals = {upsamples[0] - 1} | {j for m in neck if type(m).__name__ == 'Concat'
                                         for j in m.f if 0 <= j < upsamples[0]}
        downsamples = [m.i for m in neck if type(m).__name__ == 'Conv' and m.i > upsamples[-1]]
        self.neck_start = upsamples[0]
        self.deform = nn.ModuleDict({str(j): DeformableConv(width(layers[j])) for j in sorted(laterals)}
                                    if 'de' in parts else {})
        self.upsample = nn.ModuleDict({str(i): CARAFE(width(layers[i - 1])) for i in upsamples}
                                      if 'ca' in parts else {})
        self.attend = nn.ModuleDict({str(i): SpaceAttention() for i in downsamples} if 'pa' in parts else {})
        self.balance = (BalancedPyramid([head.cv2[i][0].conv.in_channels for i in range(head.nl)]) if 'bs' in parts
                        else None)

    def lateral_maps(self, outputs):
        """The backbone layers' outputs as the neck reads them: through the deformable convs (de), if on."""
        return [self.deform[str(j)](x) if str(j) in self.deform else x for j, x in enumerate(outputs)]

    def layer(self, m, x):
        """Neck layer m on its input x, with the parts at its place: CARAFE instead of an Upsample, the space
        attention before a bottom-up Conv."""
        i = str(m.i)
        if i in self.upsample:
            return self.upsample[i](x)
        return m(self.attend[i](x) if i in self.attend else x)

    def head_inputs(self, maps):
        """The Detect head's input maps (P3, P4, P5), balanced if bs is on."""
        return self.balance(maps) if self.balance is not None else maps
