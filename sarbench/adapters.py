"""LoRA and MoE-LoRA adapters for the attention layers of a frozen backbone."""
from torch import nn


class LoRA(nn.Module):
    """y = W x + (alpha / r) * B A x, where W is the wrapped layer (frozen by add_adapters)."""

    def __init__(self, base: nn.Linear, rank=16, alpha=32):
        super().__init__()
        self.base = base
        self.down = nn.Linear(base.in_features, rank, bias=False)  # A
        self.up = nn.Linear(rank, base.out_features, bias=False)  # B
        nn.init.zeros_(self.up.weight)  # B = 0: training starts from exactly the pretrained layer
        self.scale = alpha / rank

    # As nn.Linear's, so another adapter can wrap this one (the frozen SSL adapters of simmim.load_ssl)
    @property
    def in_features(self):
        return self.base.in_features

    @property
    def out_features(self):
        return self.base.out_features

    def forward(self, x):
        return self.base(x) + self.scale * self.up(self.down(x))


class MoELoRA(nn.Module):
    """y = W x + (alpha / r) * sum_e g_e(x) B_e A_e x, with gates g(x) = softmax(router[task](x)) per token.

    The experts' A_e are stacked in `down` and their B_e side by side in `up`, so gating each
    expert's r hidden units and applying `up` sums the experts.
    """

    def __init__(self, base: nn.Linear, experts=4, rank=4, alpha=8, tasks=2):
        super().__init__()
        self.base = base
        self.down = nn.Linear(base.in_features, experts * rank, bias=False)  # A_1 ... A_E
        self.up = nn.Linear(experts * rank, base.out_features, bias=False)  # B_1 ... B_E
        nn.init.zeros_(self.up.weight)
        self.routers = nn.ModuleList(nn.Linear(base.in_features, experts, bias=False) for _ in range(tasks))
        self.rank = rank
        self.scale = alpha / rank
        self.task = 0  # which router to use; see set_task

    def forward(self, x):
        gates = self.routers[self.task](x).softmax(dim=-1)
        gates = gates.repeat_interleave(self.rank, dim=-1)  # expert e's gate on each of its r hidden units
        return self.base(x) + self.scale * self.up(self.down(x) * gates)


def add_adapters(backbone, kind, rank=None):
    """Freeze the backbone, then wrap attn.qkv and attn.proj of every block (kind: 'lora' or 'moelora').
    rank: of each adapter (MoE-LoRA: of each expert); None keeps the defaults. alpha is 2 x rank, as in the defaults,
    so alpha / r stays 2."""
    adapter = {'lora': LoRA, 'moelora': MoELoRA}[kind]
    options = {} if rank is None else {'rank': rank, 'alpha': 2 * rank}
    backbone.requires_grad_(False)
    for blk in backbone.blocks:
        blk.attn.qkv = adapter(blk.attn.qkv, **options)
        blk.attn.proj = adapter(blk.attn.proj, **options)


def unfreeze_blocks(backbone, indices):
    """Make backbone.blocks[i] fully trainable for each i in `indices`, after add_adapters (their adapters stay).
    Returns the newly trainable parameters, the pretrained weights, so they can get their own learning rate."""
    frozen = [p for i in indices for p in backbone.blocks[i].parameters() if not p.requires_grad]
    for p in frozen:
        p.requires_grad_(True)
    return frozen


def set_task(module, task):
    """Route every MoELoRA inside `module` through the router of `task` (0 = classification, 1 = detection)."""
    for layer in module.modules():
        if isinstance(layer, MoELoRA):
            layer.task = task
