"""End-to-end runs of ``nano_moe.main.main`` on a slice of the local ``input.txt``.

These used to wrap each run in ``try/except`` and print the error, so pytest
reported them as passing even when every run failed. They now fail on error.
"""

import math
import os

import pytest

from nano_moe.config import TrainingConfig
from nano_moe.main import main

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT = os.path.join(ROOT, "input.txt")


def tiny_cfg(tmp_path, model_type, **kw):
    cfg = TrainingConfig(
        model_type=model_type, epochs=1, dataset_type="text", dataset_names=[INPUT],
        max_chars=20_000, batch_size=4, seq_len=32, feature_dim=32, hidden_dim=64,
        depth=1, n_heads=4, n_freqs=64, max_steps_per_epoch=3, max_eval_batches=2,
        save_dir=str(tmp_path),
    )
    for k, v in kw.items():
        setattr(cfg, k, v)
    return cfg


@pytest.mark.parametrize("model_type,kw", [
    ("moe", {}),
    ("moe", {"text_moe": True}),
    ("universal_phase", {}),
    ("true_phase", {}),
])
def test_main_text_pipeline(tmp_path, model_type, kw):
    model, history = main(tiny_cfg(tmp_path, model_type, **kw))
    assert len(history["step_losses"]) == 3
    assert all(math.isfinite(v) for v in history["step_losses"] + history["val_loss"])
    assert (tmp_path / "models" / "last.pt").exists()
    assert (tmp_path / "data" / "training_metrics.json").exists()


@pytest.mark.skipif(os.environ.get("NANO_MOE_NETWORK_TESTS") != "1",
                    reason="downloads ARC data; set NANO_MOE_NETWORK_TESTS=1 to run")
def test_quantum_diffusion(tmp_path):
    # main() loads the (local) text data first, then dispatches to the ARC loop.
    main(tiny_cfg(tmp_path, "quantum_diffusion"))
