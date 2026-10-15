"""Save and restore model/optimizer state together with run metadata."""

import dataclasses
from pathlib import Path
from typing import Any, Dict, Optional

import torch


def save_checkpoint(path, model, optimizer=None, cfg=None, **extra) -> Path:
    """Write ``model`` (and optionally ``optimizer``) state to ``path``.

    ``cfg`` may be a dataclass (stored as a dict) and ``extra`` holds any
    other picklable metadata such as the step or the tokenizer vocabulary.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if cfg is not None and dataclasses.is_dataclass(cfg):
        cfg = dataclasses.asdict(cfg)
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "cfg": cfg,
        "extra": extra,
    }, path)
    return path


def load_checkpoint(path, model, optimizer=None, map_location="cpu") -> Dict[str, Any]:
    """Load a checkpoint written by :func:`save_checkpoint` into ``model``.

    Loading is strict, so a checkpoint from a different architecture fails
    loudly instead of silently leaving weights uninitialised. Returns the
    stored ``cfg`` and ``extra`` metadata.
    """
    # weights_only=False: the checkpoint also stores plain-Python metadata
    # (config dict, vocabulary); only load checkpoints you created.
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and ckpt.get("optimizer") is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    return {"cfg": ckpt.get("cfg"), "extra": ckpt.get("extra", {})}
