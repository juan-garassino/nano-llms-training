"""Fast CPU unit tests for the LM pipeline, MoE routing and checkpoints."""

import math

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from nano_moe.datasets.text import CharTokenizer, TextDataset
from nano_moe.models.moe import IntegratedMoE, MoEFeedForward, load_balancing_loss
from nano_moe.models.transformer import SimpleTransformer
from nano_moe.training.checkpoint import load_checkpoint, save_checkpoint
from nano_moe.training.trainer import eval_model, train_epoch


def tiny_lm(num_experts=0, vocab=20, max_len=16):
    torch.manual_seed(0)
    return SimpleTransformer(vocab, d_model=32, n_heads=4, n_layers=2, d_ff=64,
                             max_len=max_len, dropout=0.0, num_experts=num_experts, expert_k=2)


def test_text_dataset_chunks_are_full_length():
    for n in (64, 65, 70):  # 64 used to yield a short final chunk that broke batching
        ds = TextDataset(torch.arange(n), 16)
        assert len(ds) == (n - 1) // 16
        for i in range(len(ds)):
            x, y = ds[i]
            assert x.shape == y.shape == (16,)
            assert torch.equal(y, x + 1)


def test_char_tokenizer_roundtrip():
    tok = CharTokenizer("hello world")
    assert tok.decode(tok.encode("low old")) == "low old"
    assert tok.vocab_size == len(set("hello world"))


@pytest.mark.parametrize("num_experts", [0, 4])
def test_lm_is_causal(num_experts):
    model = tiny_lm(num_experts).eval()
    a = torch.randint(0, 20, (1, 12))
    b = a.clone()
    b[0, 8] = (b[0, 8] + 1) % 20
    with torch.no_grad():
        la, lb = model(a)[0], model(b)[0]
    assert torch.allclose(la[0, :8], lb[0, :8], atol=1e-5)
    assert not torch.allclose(la[0, 8:], lb[0, 8:])


@pytest.mark.parametrize("num_experts", [0, 4])
def test_generate_keeps_tokens_beyond_context(num_experts):
    model = tiny_lm(num_experts, max_len=16)
    out = model.generate([1, 2, 3], max_new_tokens=30)
    assert out.shape == (33,)
    assert out[:3].tolist() == [1, 2, 3]


def test_load_balancing_loss_bounds():
    E, N = 4, 400
    uniform = torch.full((N, E), 1.0 / E)
    balanced_idx = torch.arange(N).remainder(E).unsqueeze(1)
    assert load_balancing_loss(uniform, balanced_idx, E).item() == pytest.approx(1.0)
    collapsed = torch.zeros(N, E)
    collapsed[:, 0] = 1.0
    collapsed_idx = torch.zeros(N, 1, dtype=torch.long)
    assert load_balancing_loss(collapsed, collapsed_idx, E).item() == pytest.approx(E)


def test_moe_ffn_routes_and_gets_gradients():
    torch.manual_seed(0)
    moe = MoEFeedForward(16, 32, num_experts=4, k=2)
    x = torch.randn(2, 50, 16)
    y = moe(x)
    assert y.shape == x.shape
    assert moe.expert_usage.sum().item() == pytest.approx(1.0)
    (y.pow(2).mean() + moe.aux_loss).backward()
    assert moe.router.weight.grad is not None and moe.router.weight.grad.abs().sum() > 0
    used = moe.expert_usage > 0
    for e, expert in enumerate(moe.experts):
        if used[e]:
            assert expert[0].weight.grad.abs().sum() > 0


def test_aux_loss_reduces_routing_imbalance():
    """Optimising only the load-balancing loss should spread a collapsed router."""
    torch.manual_seed(0)
    moe = MoEFeedForward(8, 16, num_experts=4, k=1)
    with torch.no_grad():
        moe.router.weight.normal_(0, 0.1)
        moe.router.weight[0] += 1.0  # strongly prefer expert 0 for positive inputs
    x = torch.rand(256, 8)
    opt = torch.optim.SGD(moe.router.parameters(), lr=1.0)
    moe(x)
    start = moe.aux_loss.item()
    for _ in range(100):
        opt.zero_grad()
        moe(x)
        moe.aux_loss.backward()
        opt.step()
    moe(x)
    assert start > 3.0
    assert moe.aux_loss.item() < start
    assert moe.expert_usage.max().item() < 0.6


def lm_loaders(n=32, L=16, vocab=20):
    torch.manual_seed(0)
    x = torch.randint(0, vocab, (n, L))
    y = torch.roll(x, -1, 1)
    ds = TensorDataset(x, y)
    return {"t": {"train_loader": DataLoader(ds, batch_size=8), "test_loader": DataLoader(ds, batch_size=8)}}


@pytest.mark.parametrize("num_experts", [0, 4])
def test_train_and_eval_with_lm(tmp_path, num_experts):
    from nano_moe.training.tracker import ExperimentTracker
    model = tiny_lm(num_experts)
    loaders = lm_loaders()
    opt = torch.optim.Adam(model.parameters(), 3e-3)
    losses, gates, _ = train_epoch(model, loaders, opt, nn.CrossEntropyLoss(), "cpu", ["t"],
                                   ExperimentTracker(str(tmp_path)), 1, max_steps=3)
    assert len(losses) == 3 and all(math.isfinite(l) for l in losses)
    if num_experts:
        assert gates[0].shape == (num_experts,)
    val = {}
    accs = eval_model(model, loaders, "cpu", ["t"], max_batches=2, losses=val)  # used to raise ValueError
    assert 0.0 <= accs["t"] <= 1.0
    assert math.isfinite(val["t"])


@pytest.mark.parametrize("num_experts", [0, 4])
def test_checkpoint_roundtrip(tmp_path, num_experts):
    model = tiny_lm(num_experts).eval()
    opt = torch.optim.Adam(model.parameters())
    path = save_checkpoint(tmp_path / "ck.pt", model, opt, step=7)
    fresh = tiny_lm(num_experts)
    for p in fresh.parameters():
        nn.init.zeros_(p)
    meta = load_checkpoint(path, fresh, torch.optim.Adam(fresh.parameters()))
    fresh.eval()
    x = torch.randint(0, 20, (2, 10))
    with torch.no_grad():
        assert torch.allclose(model(x)[0], fresh(x)[0])
    assert meta["extra"]["step"] == 7


def test_checkpoint_rejects_wrong_architecture(tmp_path):
    path = save_checkpoint(tmp_path / "ck.pt", tiny_lm(0))
    with pytest.raises(RuntimeError):
        load_checkpoint(path, tiny_lm(4))


@pytest.mark.parametrize("k", [1, 2])
def test_integrated_moe_gates(k):
    torch.manual_seed(0)
    model = IntegratedMoE(num_experts=4, feature_dim=32, hidden_dim=32, num_classes=10, k=k)
    x = torch.randn(4, 1, 28, 28)
    for train in (True, False):
        model.train(train)
        out, aux, mean_gate, gate, _, _ = model(x, return_attention=True)
        assert out.shape == (4, 10)
        assert torch.allclose(gate.sum(1), torch.ones(4), atol=1e-5)
        assert (gate > 0).sum(1).max() <= (4 if (train and k == 1) else k)
        assert 0.0 < aux.item() <= 4.0 + 1e-5
    model.train()
    out, aux, *_ = model(x)
    (out.sum() + aux).backward()
    assert model.router[0].weight.grad.abs().sum() > 0
