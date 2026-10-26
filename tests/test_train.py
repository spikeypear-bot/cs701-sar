"""train.py's weight EMA and optimizer options, on small CPU modules."""
import math
from types import SimpleNamespace

import torch
from torch import nn

from train import EMA, make_optimizer


def test_ema_follows_the_weights_and_batchnorm_statistics_with_a_ramped_decay():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(3, 3), nn.BatchNorm1d(3))
    ema = EMA(model, decay=0.9, tau=1)
    start = {k: v.clone() for k, v in ema.model.state_dict().items()}
    with torch.no_grad():
        for p in model.parameters():
            p.add_(1)
    model.train()(torch.randn(8, 3))  # moves the running statistics
    ema.update(model)
    d = 0.9 * (1 - math.exp(-1))
    trained = model.state_dict()
    for k, v in ema.model.state_dict().items():
        if v.dtype.is_floating_point:
            assert torch.allclose(v, d * start[k] + (1 - d) * trained[k])
    assert ema.model.state_dict()['1.num_batches_tracked'] == 0 and trained['1.num_batches_tracked'] == 1
    assert not any(p.requires_grad for p in ema.model.parameters()) and not ema.model.training


def test_sgd_drops_weight_decay_on_1d_parameters_and_keeps_the_head_group_second():
    backbone, head = nn.Linear(4, 4), nn.Sequential(nn.LayerNorm(4), nn.Linear(4, 2))
    groups = [{'params': list(backbone.parameters()), 'lr': 0.02}, {'params': list(head.parameters())}]
    optimizer = make_optimizer(groups, SimpleNamespace(optimizer='sgd', lr=0.01, momentum=0.937, weight_decay=5e-4))
    g = optimizer.param_groups
    assert isinstance(optimizer, torch.optim.SGD) and [len(group['params']) for group in g] == [1, 1, 1, 3]
    assert g[1]['params'][0] is head[1].weight and g[1]['lr'] == 0.01 and g[1]['weight_decay'] == 5e-4
    assert g[2]['lr'] == 0.02 and g[2]['weight_decay'] == 0 and g[3]['weight_decay'] == 0
    assert g[0]['nesterov'] and g[0]['momentum'] == 0.937
    adamw = make_optimizer([{'params': list(backbone.parameters())}],
                           SimpleNamespace(optimizer='adamw', lr=1e-3, weight_decay=5e-4))
    assert isinstance(adamw, torch.optim.AdamW) and adamw.param_groups[0]['weight_decay'] == 5e-4
