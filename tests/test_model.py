import copy
import json
from pathlib import Path
import torch
from next_moe.model import ModelConfig, NextMoE, delta_chunk, Engram

torch.set_num_threads(1)


def recurrent(q, k, v, log_alpha, beta, state):
    ys = []
    for t in range(q.shape[-2]):
        state = log_alpha[..., t, None, None].exp() * state
        prediction = torch.einsum('bhk,bhkv->bhv', k[..., t, :], state)
        error = v[..., t, :] - prediction
        state = state + beta[..., t, None, None] * k[..., t, :, None] * error[..., None, :]
        ys.append(torch.einsum('bhk,bhkv->bhv', q[..., t, :], state))
    return torch.stack(ys, -2), state


def test_chunk_delta_forward_and_gradients():
    torch.manual_seed(7)
    values = [torch.nn.functional.normalize(torch.randn(2, 3, 7, 5), dim=-1),
              torch.nn.functional.normalize(torch.randn(2, 3, 7, 5), dim=-1),
              torch.randn(2, 3, 7, 4), -torch.rand(2, 3, 7),
              torch.rand(2, 3, 7), torch.randn(2, 3, 5, 4)]
    a = [v.clone().requires_grad_() for v in values]
    b = [v.clone().requires_grad_() for v in values]
    y1, s1 = delta_chunk(*a)
    y2, s2 = recurrent(*b)
    torch.testing.assert_close(y1, y2, atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(s1, s2, atol=2e-6, rtol=2e-5)
    (y1.square().sum() + s1.square().sum()).backward()
    (y2.square().sum() + s2.square().sum()).backward()
    for x, y in zip(a, b):
        torch.testing.assert_close(x.grad, y.grad, atol=2e-5, rtol=2e-4)


def test_causal_prefix_and_chunk_invariance():
    torch.manual_seed(11)
    cfg = ModelConfig.tiny(); cfg.checkpoint_blocks = False
    model = NextMoE(cfg).eval()
    tokens = torch.randint(4, cfg.vocab_size, (2, 11))
    changed = tokens.clone(); changed[:, 7:] = torch.randint(4, cfg.vocab_size, (2, 4))
    with torch.no_grad():
        full = model(tokens)
        future = model(changed)
        prefix = model(tokens[:, :7])
        model.config.gdn_chunk = 3
        chunked = model(tokens)
    torch.testing.assert_close(full[:, :7], future[:, :7], atol=2e-5, rtol=2e-4)
    torch.testing.assert_close(full[:, :7], prefix, atol=2e-5, rtol=2e-4)
    torch.testing.assert_close(full, chunked, atol=2e-5, rtol=2e-4)


def test_checkpoint_and_shared_kv_gradients():
    torch.manual_seed(19)
    cfg = ModelConfig.tiny(); cfg.checkpoint_blocks = False
    baseline = NextMoE(cfg)
    other = copy.deepcopy(baseline); other.config.checkpoint_blocks = True
    x = torch.randint(4, cfg.vocab_size, (1, 9)); y = torch.randint(4, cfg.vocab_size, (1, 9))
    for model in (baseline, other):
        out = model(x, y, loss_chunk=3)
        (out['ce'] + .01 * out['aux']).backward()
    for (name, p), (_, q) in zip(baseline.named_parameters(), other.named_parameters()):
        if p.grad is None:
            assert q.grad is None, name
        else:
            assert torch.isfinite(q.grad).all(), name
            torch.testing.assert_close(p.grad, q.grad, atol=3e-5, rtol=3e-4)
    source, reader = baseline.blocks[3], baseline.blocks[7]
    assert reader.mixer.k is None and reader.mixer.v is None
    x = torch.randn(1, 7, cfg.dim, requires_grad=True)
    _, kv = source.mixer(x)
    # Receiver-only loss must backpropagate to the source's unique K/V.
    baseline.zero_grad(set_to_none=True)
    receiver, _ = reader.mixer(torch.randn_like(x), kv)
    receiver.square().sum().backward()
    assert source.mixer.k.weight.grad.abs().sum() > 0
    assert source.mixer.v.weight.grad.abs().sum() > 0


def test_engram_causal_reset_and_single_token():
    cfg = ModelConfig.tiny()
    memory = Engram(cfg)
    a = torch.tensor([[11, 12, 2, 15, 16]])
    b = torch.tensor([[21, 22, 2, 15, 16]])
    for aa, bb in zip(memory.addresses(a), memory.addresses(b)):
        assert torch.equal(aa[:, 2:], bb[:, 2:])
    assert all(z.shape == (1, 1) for z in memory.addresses(torch.tensor([[11]])))


def test_full_profile_parameter_count():
    with torch.device('meta'):
        model = NextMoE(ModelConfig())
    family = json.loads((Path(__file__).parents[1] / 'configs/model-family.json').read_text())
    proposed = family['profiles'][0]['estimate']
    total = sum(p.numel() for p in model.parameters())
    # Three Q norms + two owner K norms were excluded as small mixer gains.
    assert total == proposed['total_parameters'] + 5 * 64
    table = sum(p.numel() for block in model.blocks if block.memory
                for p in block.memory.tables.parameters())
    assert table == proposed['ngram_table_parameters']


def test_tiny_model_can_overfit():
    torch.manual_seed(41)
    cfg = ModelConfig.tiny(); cfg.checkpoint_blocks = True
    model = NextMoE(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.008)
    x = torch.tensor([[4, 5, 6, 7, 4, 5, 6, 7]])
    y = torch.tensor([[5, 6, 7, 4, 5, 6, 7, 4]])
    before = float(model(x, y)['ce'].detach())
    for _ in range(25):
        optimizer.zero_grad()
        out = model(x, y)
        (out['ce'] + .01 * out['aux']).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
    after = float(model(x, y)['ce'].detach())
    assert after < before * .5, (before, after)
