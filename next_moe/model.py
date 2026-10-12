"""Trainable v0.3 reference: GR + causal Engram + GDN/GQA + no-drop MoE.

The chunkwise GDN is an exact triangular reformulation of the recurrence,
not a production Triton kernel. Generation recomputes the prefix (no KV API).
"""
from dataclasses import asdict, dataclass, field
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


@dataclass
class ModelConfig:
    vocab_size: int = 65536
    layers: int = 12
    dim: int = 512
    experts: int = 16
    top_k: int = 2
    expert_dim: int = 448
    branches: int = 4
    rank: int = 64
    q_heads: int = 8
    kv_heads: int = 2
    head_dim: int = 64
    gdn_k_heads: int = 4
    gdn_v_heads: int = 8
    gdn_key_dim: int = 64
    gdn_value_dim: int = 64
    table_rows: list[int] = field(default_factory=lambda: [390647, 390653, 390671, 390673,
                                                          390703, 390707, 390721, 390727])
    row_dim: int = 32
    full_every: int = 4
    kv_group: int = 2
    memory_layer: int = 1  # zero-based: decoder block 2
    gdn_chunk: int = 32
    rope_theta: float = 10000.0
    eos_id: int = 2
    pad_id: int = 0
    checkpoint_blocks: bool = True

    def __post_init__(self):
        if self.layers % self.full_every or not 0 <= self.memory_layer < self.layers:
            raise ValueError('Layers must end with a full layer; memory layer must exist.')
        if self.q_heads % self.kv_heads or self.gdn_v_heads % self.gdn_k_heads:
            raise ValueError('Head ratios must be integral.')
        if not 1 <= self.top_k <= self.experts or len(self.table_rows) != 8:
            raise ValueError('Invalid expert or Engram configuration.')
        if self.kv_group != 2 or self.head_dim % 2 or self.gdn_chunk < 1:
            raise ValueError('This reference supports KV pairs and even RoPE dimensions.')

    @classmethod
    def tiny(cls):
        """Only for correctness tests; not the 0.2B+0.1B profile."""
        return cls(vocab_size=512, layers=8, dim=32, experts=4, top_k=2,
                   expert_dim=24, rank=4, q_heads=4, kv_heads=2, head_dim=8,
                   gdn_k_heads=2, gdn_v_heads=4, gdn_key_dim=8, gdn_value_dim=8,
                   table_rows=[101, 103, 107, 109, 113, 127, 131, 137], row_dim=4,
                   gdn_chunk=4)

    def to_dict(self):
        return asdict(self)


class RMSNorm(nn.Module):
    def __init__(self, shape, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(shape))
        self.eps = eps

    def forward(self, x):
        y = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.eps)
        return (y * self.weight.float()).to(x.dtype)


class GatedResidual(nn.Module):
    def __init__(self, cfg, write=True):
        super().__init__()
        self.n, self.d = cfg.branches, cfg.dim
        self.norm = RMSNorm((self.n, self.d))
        self.down = nn.Linear(self.n * self.d, cfg.rank, bias=False)
        self.up = nn.Linear(cfg.rank, self.n * self.d, bias=False)
        self.write = nn.Linear(self.n * self.d, self.n, bias=False) if write else None

    def read(self, r):
        normalized = self.norm(r)
        z = normalized.flatten(-2)
        gate = self.up(F.silu(self.down(z) / self.n)).sigmoid().reshape_as(r)
        x = (gate * normalized).mean(-2)
        s = 2 * (self.write(z) / self.n).sigmoid() if self.write is not None else None
        return x, s

    @staticmethod
    def add(r, y, s):
        return r + s.unsqueeze(-1).to(r.dtype) * y.unsqueeze(-2).to(r.dtype)


class CausalConv(nn.Module):
    def __init__(self, width, kernel=4, dilation=1):
        super().__init__()
        self.left = (kernel - 1) * dilation
        self.conv = nn.Conv1d(width, width, kernel, groups=width,
                              dilation=dilation, bias=False)

    def forward(self, x):
        return self.conv(F.pad(x.transpose(1, 2), (self.left, 0))).transpose(1, 2)


class Engram(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tables = nn.ModuleList([nn.Embedding(rows, cfg.row_dim) for rows in cfg.table_rows])
        m, n, d = 8 * cfg.row_dim, cfg.branches, cfg.dim
        self.value = nn.Linear(m, d, bias=False)
        self.key = nn.Linear(m, n * d, bias=False)
        self.query_norm, self.key_norm, self.conv_norm = [RMSNorm((n, d)) for _ in range(3)]
        self.conv = CausalConv(n * d, kernel=4, dilation=3)

    def addresses(self, tokens):
        # Token IDs are kept unchanged: no accent stripping or case folding.
        # Reset suffix lookup at EOS; left padding is deterministic at sample starts.
        lag1 = F.pad(tokens[:, :-1], (1, 0), value=self.cfg.pad_id)
        lag2 = F.pad(tokens[:, :-2], (2, 0), value=self.cfg.pad_id)[:, :tokens.shape[1]]
        reset = tokens.eq(self.cfg.eos_id) | lag1.eq(self.cfg.eos_id)
        lag1 = torch.where(reset, self.cfg.pad_id, lag1)
        lag2 = torch.where(reset | lag2.eq(self.cfg.eos_id), self.cfg.pad_id, lag2)
        hashes = []
        for i, rows in enumerate(self.cfg.table_rows):
            # Small integer coefficients keep arithmetic far below int64 overflow.
            code = (tokens + 1) * (1009 + 104 * i) + (lag1 + 1) * (9176 + 62 * i)
            if i >= 4:
                code = code + (lag2 + 1) * (65537 + 50 * i)
            hashes.append(code.remainder(rows))
        return hashes

    def forward(self, tokens, r):
        e = torch.cat([table(ids) for table, ids in zip(self.tables, self.addresses(tokens))], -1)
        e = e.to(r.dtype)
        v = self.value(e)
        key = self.key(e).reshape_as(r)
        score = (self.query_norm(r).float() * self.key_norm(key).float()).sum(-1) / math.sqrt(self.cfg.dim)
        transformed = score.sign() * score.abs().clamp_min(1e-6).sqrt()
        u = transformed.sigmoid().unsqueeze(-1).to(r.dtype) * v.unsqueeze(-2)
        conv = self.conv(self.conv_norm(u).flatten(-2)).reshape_as(r)
        return r + u.to(r.dtype) + F.silu(conv).to(r.dtype)


def delta_chunk(q, k, v, log_alpha, beta, state):
    """Exact gated-delta recurrence within one chunk. Shapes B,H,T,D; state B,H,K,V.

    Solve write residuals against a unit-lower-triangular matrix. FP32 avoids
    accumulation loss; no attention across future positions is performed.
    """
    with torch.autocast(device_type=q.device.type, enabled=False):
        q, k, v, log_alpha, beta, state = [x.float() for x in (q, k, v, log_alpha, beta, state)]
        length = q.shape[-2]
        cumulative = log_alpha.cumsum(-1)
        decay = cumulative.exp()
        ratios = (cumulative.unsqueeze(-1) - cumulative.unsqueeze(-2)).clamp_max(0).exp()
        kk = k @ k.transpose(-1, -2)
        lower = (ratios * kk * beta.unsqueeze(-2)).tril(-1)
        matrix = lower + torch.eye(length, device=q.device, dtype=torch.float32)
        rhs = v - decay.unsqueeze(-1) * (k @ state)
        errors = torch.linalg.solve_triangular(matrix, rhs, upper=False, unitriangular=True)
        weights = (ratios * (q @ k.transpose(-1, -2)) * beta.unsqueeze(-2)).tril()
        y = decay.unsqueeze(-1) * (q @ state) + weights @ errors
        last_ratios = (cumulative[..., -1:] - cumulative).exp()
        next_state = decay[..., -1, None, None] * state + k.transpose(-1, -2) @ (
            errors * (last_ratios * beta).unsqueeze(-1))
        return y, next_state


class GDN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        k = cfg.gdn_k_heads * cfg.gdn_key_dim
        w = cfg.gdn_v_heads * cfg.gdn_value_dim
        self.qkv = nn.Linear(cfg.dim, 2 * k + w, bias=False)
        self.conv = CausalConv(2 * k + w)
        self.decay = nn.Linear(cfg.dim, cfg.gdn_v_heads, bias=False)
        self.beta = nn.Linear(cfg.dim, cfg.gdn_v_heads, bias=False)
        self.gate = nn.Linear(cfg.dim, w, bias=False)
        self.out = nn.Linear(w, cfg.dim, bias=False)
        self.norm = RMSNorm(cfg.gdn_value_dim)
        self.A_log = nn.Parameter(torch.linspace(0, math.log(16), cfg.gdn_v_heads))
        # Initial update time constants in [0.001, 0.1].
        dt = torch.exp(torch.linspace(math.log(.001), math.log(.1), cfg.gdn_v_heads))
        self.dt_bias = nn.Parameter(torch.log(torch.expm1(dt)))

    def forward(self, x):
        c = self.cfg
        b, t, _ = x.shape
        kw, vw = c.gdn_k_heads * c.gdn_key_dim, c.gdn_v_heads * c.gdn_value_dim
        q, k, v = F.silu(self.conv(self.qkv(x))).split([kw, kw, vw], -1)
        repeat = c.gdn_v_heads // c.gdn_k_heads
        q = q.reshape(b, t, c.gdn_k_heads, c.gdn_key_dim).repeat_interleave(repeat, dim=2).transpose(1, 2)
        k = k.reshape(b, t, c.gdn_k_heads, c.gdn_key_dim).repeat_interleave(repeat, dim=2).transpose(1, 2)
        q, k = F.normalize(q.float(), dim=-1, eps=1e-6), F.normalize(k.float(), dim=-1, eps=1e-6)
        v = v.reshape(b, t, c.gdn_v_heads, c.gdn_value_dim).transpose(1, 2)
        log_alpha = -F.softplus(self.decay(x).float() + self.dt_bias.float()) * self.A_log.float().exp()
        log_alpha = log_alpha.transpose(1, 2)
        beta = self.beta(x).float().sigmoid().transpose(1, 2)
        state = torch.zeros(b, c.gdn_v_heads, c.gdn_key_dim, c.gdn_value_dim, device=x.device)
        outputs = []
        for start in range(0, t, c.gdn_chunk):
            stop = min(start + c.gdn_chunk, t)
            y, state = delta_chunk(q[:, :, start:stop], k[:, :, start:stop], v[:, :, start:stop],
                                  log_alpha[:, :, start:stop], beta[:, :, start:stop], state)
            outputs.append(y)
        y = torch.cat(outputs, dim=2).transpose(1, 2).to(x.dtype)
        y = self.norm(y).reshape(b, t, vw) * self.gate(x).sigmoid()
        return self.out(y)


def rope(x, theta):
    t, d = x.shape[-2:]
    positions = torch.arange(t, device=x.device).float()
    inverse = theta ** (-torch.arange(0, d, 2, device=x.device).float() / d)
    angle = positions[:, None] * inverse[None, :]
    cos, sin = angle.cos().to(x.dtype), angle.sin().to(x.dtype)
    left, right = x[..., :d // 2], x[..., d // 2:]
    return torch.cat([left * cos - right * sin, right * cos + left * sin], -1)


class GQA(nn.Module):
    def __init__(self, cfg, owner):
        super().__init__()
        self.cfg, self.owner = cfg, owner
        self.q = nn.Linear(cfg.dim, cfg.q_heads * cfg.head_dim, bias=False)
        self.gate = nn.Linear(cfg.dim, cfg.q_heads * cfg.head_dim, bias=False)
        self.out = nn.Linear(cfg.q_heads * cfg.head_dim, cfg.dim, bias=False)
        self.q_norm = RMSNorm(cfg.head_dim)
        self.k_norm = RMSNorm(cfg.head_dim) if owner else None
        self.k = nn.Linear(cfg.dim, cfg.kv_heads * cfg.head_dim, bias=False) if owner else None
        self.v = nn.Linear(cfg.dim, cfg.kv_heads * cfg.head_dim, bias=False) if owner else None

    def forward(self, x, kv=None):
        c = self.cfg
        b, t, _ = x.shape
        q = rope(self.q_norm(self.q(x).reshape(b, t, c.q_heads, c.head_dim)).transpose(1, 2), c.rope_theta)
        if self.owner:
            k = rope(self.k_norm(self.k(x).reshape(b, t, c.kv_heads, c.head_dim)).transpose(1, 2), c.rope_theta)
            v = self.v(x).reshape(b, t, c.kv_heads, c.head_dim).transpose(1, 2)
            kv = (k, v)
        if kv is None:
            raise ValueError('KV reader requires source tensors.')
        # Explicit repeat is portable to Colab SDPA backends; logical KV stays shared.
        k, v = [z.repeat_interleave(c.q_heads // c.kv_heads, dim=1) for z in kv]
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=0.0)
        y = y.transpose(1, 2).reshape(b, t, c.q_heads * c.head_dim)
        return self.out(y * self.gate(x).sigmoid()), kv


class Expert(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.gate = nn.Linear(dim, hidden, bias=False)
        self.up = nn.Linear(dim, hidden, bias=False)
        self.down = nn.Linear(hidden, dim, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class MoE(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.router = nn.Linear(cfg.dim, cfg.experts, bias=False)
        self.experts = nn.ModuleList([Expert(cfg.dim, cfg.expert_dim) for _ in range(cfg.experts)])
        self.shared = Expert(cfg.dim, cfg.expert_dim)

    def forward(self, x):
        shape = x.shape
        flat = x.reshape(-1, self.cfg.dim)
        with torch.autocast(device_type=x.device.type, enabled=False):
            probabilities = F.linear(flat.float(), self.router.weight.float()).softmax(-1)
        weights, selected = probabilities.topk(self.cfg.top_k, dim=-1)
        weights = weights / weights.sum(-1, keepdim=True)
        y = torch.zeros_like(flat)
        for idx, expert in enumerate(self.experts):
            token, slot = torch.where(selected == idx)
            if token.numel():
                out = expert(flat[token]) * weights[token, slot, None].to(x.dtype)
                y = y.index_add(0, token, out.to(y.dtype))
        counts = torch.bincount(selected.flatten(), minlength=self.cfg.experts).float()
        fraction = counts / selected.numel()
        auxiliary = self.cfg.experts * (fraction.detach() * probabilities.mean(0)).sum()
        return (y + self.shared(flat)).reshape(shape), auxiliary, counts.detach()


class Block(nn.Module):
    def __init__(self, cfg, index):
        super().__init__()
        self.full = (index + 1) % cfg.full_every == 0
        self.owner = self.full and ((index + 1) // cfg.full_every - 1) % cfg.kv_group == 0
        self.memory = Engram(cfg) if index == cfg.memory_layer else None
        self.attn_gr, self.moe_gr = GatedResidual(cfg), GatedResidual(cfg)
        self.mixer = GQA(cfg, self.owner) if self.full else GDN(cfg)
        self.moe = MoE(cfg)

    def forward(self, r, tokens, kv):
        if self.memory is not None:
            r = self.memory(tokens, r)
        x, s = self.attn_gr.read(r)
        if self.full:
            y, kv = self.mixer(x, kv)
        else:
            y = self.mixer(x)
        r = self.attn_gr.add(r, y, s)
        x, s = self.moe_gr.read(r)
        y, aux, counts = self.moe(x)
        return self.moe_gr.add(r, y, s), kv, aux, counts


class NextMoE(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.config = cfg
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.blocks = nn.ModuleList([Block(cfg, i) for i in range(cfg.layers)])
        self.final_gr = GatedResidual(cfg, write=False)
        self.apply(self._init)
        # Zero initial write projections => scalar write 1. Small mixer outputs
        # keep a newly initialized multi-stream residual stable.
        for block in self.blocks:
            nn.init.zeros_(block.attn_gr.write.weight)
            nn.init.zeros_(block.moe_gr.write.weight)
            nn.init.normal_(block.mixer.out.weight, std=.02 / math.sqrt(2 * cfg.layers))
        self.last_counts = None

    @staticmethod
    def _init(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=.02)

    def features(self, tokens):
        r = self.embedding(tokens)
        if torch.is_autocast_enabled(tokens.device.type):
            r = r.to(torch.get_autocast_dtype(tokens.device.type))
        r = r.unsqueeze(-2).expand(-1, -1, self.config.branches, -1).clone()
        kv, auxiliary, counts = None, [], []
        for block in self.blocks:
            if self.training and self.config.checkpoint_blocks and torch.is_grad_enabled():
                r, kv, aux, count = checkpoint(block, r, tokens, kv, use_reentrant=False)
            else:
                r, kv, aux, count = block(r, tokens, kv)
            auxiliary.append(aux)
            counts.append(count)
        self.last_counts = torch.stack(counts)
        return self.final_gr.read(r)[0], torch.stack(auxiliary).mean()

    def forward(self, tokens, targets=None, loss_chunk=64):
        hidden, auxiliary = self.features(tokens)
        if targets is None:
            return F.linear(hidden, self.embedding.weight)
        # Chunk and checkpoint the softmax/head to avoid B*T*V live logits.
        total = hidden.new_zeros((), dtype=torch.float32)
        for start in range(0, tokens.shape[1], loss_chunk):
            h = hidden[:, start:start + loss_chunk]
            y = targets[:, start:start + loss_chunk]
            def loss_fn(hh, yy):
                logits = F.linear(hh, self.embedding.weight)
                return F.cross_entropy(logits.float().reshape(-1, self.config.vocab_size),
                                       yy.reshape(-1), reduction='sum')
            loss = checkpoint(loss_fn, h, y, use_reentrant=False) if self.training else loss_fn(h, y)
            total = total + loss
        return {'ce': total / targets.numel(), 'aux': auxiliary}

    @torch.no_grad()
    def generate(self, tokens, max_new_tokens=64, temperature=.8, top_k=40, context=512, valid_vocab=None):
        was_training = self.training
        self.eval()
        try:
            for _ in range(max_new_tokens):
                hidden, _ = self.features(tokens[:, -context:])
                logits = F.linear(hidden[:, -1], self.embedding.weight).float()
                if valid_vocab is not None:
                    logits[:, valid_vocab:] = float('-inf')
                logits[:, self.config.pad_id] = float('-inf')
                if temperature <= 0:
                    nxt = logits.argmax(-1, keepdim=True)
                else:
                    logits = logits / temperature
                    threshold = logits.topk(min(top_k, logits.shape[-1]), dim=-1).values[:, -1:]
                    logits = logits.masked_fill(logits < threshold, float('-inf'))
                    nxt = torch.multinomial(logits.softmax(-1), 1)
                tokens = torch.cat([tokens, nxt], -1)
                if tokens.shape[0] == 1 and nxt.item() == self.config.eos_id:
                    break
            return tokens
        finally:
            self.train(was_training)
