"""Offline regression tests for bugs fixed by the guardian run (no network, CPU only)."""
import os
import sys

import pytest

torch = pytest.importorskip("torch")
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nano_moe.datasets.text import TextDataset  # noqa: E402
from nano_moe.training.trainer import eval_model  # noqa: E402


@pytest.mark.parametrize("n_tokens,seq_len", [(64, 8), (65, 8), (70, 8), (9, 8)])
def test_text_dataset_every_item_is_full_length(n_tokens, seq_len):
    # Bug: __len__ was len(tokens)//seq_len, so the last item could run off the
    # end and yield x/y shorter than seq_len, crashing default collate.
    ds = TextDataset(torch.arange(n_tokens), seq_len)
    assert len(ds) >= 1
    for i in range(len(ds)):
        x, y = ds[i]
        assert x.shape == (seq_len,) and y.shape == (seq_len,)
        assert torch.equal(x[1:], y[:-1])


def test_text_dataset_too_short_is_empty():
    assert len(TextDataset(torch.arange(5), 8)) == 0


class _SixTupleLM(torch.nn.Module):
    """Mimics SimpleTransformer / UniversalSFPT: forward returns a 6-tuple."""

    def __init__(self, vocab=7):
        super().__init__()
        self.emb = torch.nn.Embedding(vocab, vocab)

    def forward(self, x, dataset_idx=None, return_attention=False, classification=True):
        logits = self.emb(x)
        return logits, torch.tensor(0.0), None, None, None, {}


def test_eval_model_accepts_six_tuple_models():
    # Bug: eval_model did `logits, _, _ = model(x)` and raised
    # "too many values to unpack" for every text model (default pipeline).
    x = torch.randint(0, 7, (4, 5))
    loaders = {"t": {"test_loader": DataLoader(TensorDataset(x, x), batch_size=2)}}
    accs = eval_model(_SixTupleLM(), loaders, "cpu", ["t"])
    assert 0.0 <= accs["t"] <= 1.0


def test_main_universal_phase_with_image_data_uses_num_classes(monkeypatch):
    # Bug: universal_phase used `vocab_size`, which is only defined for text data
    # -> NameError for dataset_type="image".
    import nano_moe.main as m

    class _Stop(Exception):
        pass

    seen = {}

    def fake_model(vocab_size, **kw):
        seen["vocab_size"] = vocab_size
        raise _Stop

    monkeypatch.setattr(m, "get_multimnist_loaders", lambda names, bs, num_workers=0: ({}, {"mnist": 10}))
    monkeypatch.setattr(m, "UniversalSFPT", fake_model)
    cfg = m.TrainingConfig()
    cfg.dataset_type = "image"
    cfg.dataset_names = ["mnist"]
    cfg.model_type = "universal_phase"
    with pytest.raises(_Stop):
        m.main(cfg)
    assert seen["vocab_size"] == 10
