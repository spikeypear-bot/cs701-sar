"""Both backbones, pretrained and from scratch, map [2, 1, 512, 512] images to the final norm's [2, 1024, 768]
patch tokens; pretrained=True really loads the published weights."""
import pytest
import torch

from sarbench.backbones import build_backbone

FINAL_NORM = {'vit': 'vit.norm', 'terramind': 'terramind.encoder_norm'}


@pytest.mark.parametrize('pretrained', [True, False])
@pytest.mark.parametrize('name', ['vit', 'terramind'])
def test_patch_tokens(name, pretrained):
    backbone = build_backbone(name, pretrained).cuda().eval()
    normed = []
    backbone.get_submodule(FINAL_NORM[name]).register_forward_hook(lambda module, args, out: normed.append(out))
    with torch.no_grad():
        tokens = backbone(torch.randn(2, 1, 512, 512, device='cuda'))
    assert tokens.shape == (2, 1024, 768)  # 32 x 32 patches, no CLS token
    assert tokens.isfinite().all()
    assert torch.equal(tokens, normed[0][:, -1024:])  # the final norm's output; ViT's norm also sees the CLS token
    assert len(backbone.blocks) == 12 and backbone.embed_dim == 768


@pytest.mark.parametrize('name', ['vit', 'terramind'])
def test_pretrained_weights_are_loaded(name):
    first, second, scratch = (dict(build_backbone(name, p).named_parameters()) for p in (True, True, False))
    assert all(torch.equal(first[k], second[k]) for k in first)  # no weight left at a random init
    assert not any(torch.equal(first[k], scratch[k]) for k in first)  # nor at the scratch init


@pytest.mark.parametrize('pretrained', [True, False])
def test_convnext_maps(pretrained):
    backbone = build_backbone('convnext_b', pretrained, drop_path=0.6).cuda().eval()
    with torch.no_grad():
        maps = backbone(torch.randn(2, 1, 512, 512, device='cuda'))
    assert [m.shape for m in maps] == [(2, c, 512 // s, 512 // s)
                                       for c, s in zip((128, 256, 512, 1024), (4, 8, 16, 32))]
    assert all(m.isfinite().all() for m in maps)
    assert len(backbone.blocks) == 36 and backbone.embed_dim == 1024


def test_convnext_pretrained_weights_are_loaded():
    """Every timm weight is the published one; only the new output norms start at LayerNorm's init."""
    first, second, scratch = (dict(build_backbone('convnext_b', p).convnext.named_parameters())
                              for p in (True, True, False))
    assert all(torch.equal(first[k], second[k]) for k in first)
    assert not any(torch.equal(first[k], scratch[k]) for k in first if 'norm' not in k)  # norms init at 1 and 0


def test_convnext_trains_under_bf16():
    """As train.py runs it: bf16 autocast, drop path, MoE-LoRA, forward and backward (job 338636 failed here in cuDNN)."""
    from sarbench.adapters import add_adapters
    backbone = build_backbone('convnext_b', pretrained=False, drop_path=0.6)
    add_adapters(backbone, 'moelora')
    backbone.cuda().train()
    with torch.autocast('cuda', dtype=torch.bfloat16):
        maps = backbone(torch.randn(16, 1, 512, 512, device='cuda'))
    sum(m.float().mean() for m in maps).backward()
    # task 0's routers only: task 1's are unused in this pass
    assert all(p.grad is not None for n, p in backbone.named_parameters() if p.requires_grad and 'routers.1' not in n)
