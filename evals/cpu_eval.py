"""Short CPU evaluation of the nano LM pipeline (dense vs. MoE).

Trains two tiny character-level language models on the first ``--max-chars``
characters of ``input.txt`` through ``nano_moe.main.main`` (the same path as
``make demo-shakespeare``), then checks:

* loss decreases (mean of the last 20 train steps vs. the first 20),
* validation loss and bits-per-character,
* MoE expert usage (max load fraction, normalised routing entropy),
* checkpoint round trip (reload into a fresh model, compare logits),
* generation returns prompt + ``max_new_tokens`` tokens.

Results are printed and written to ``--out`` as JSON. Runs in a few minutes
on 4 CPU cores with the defaults.

    python evals/cpu_eval.py [--steps 300] [--out results/cpu_eval.json]
"""

import argparse
import json
import math
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nano_moe.config import TrainingConfig  # noqa: E402
from nano_moe.datasets.text import CharTokenizer  # noqa: E402
from nano_moe.main import main  # noqa: E402
from nano_moe.models.transformer import SimpleTransformer  # noqa: E402
from nano_moe.training.checkpoint import load_checkpoint  # noqa: E402


def make_cfg(args, text_moe: bool, save_dir: str) -> TrainingConfig:
    return TrainingConfig(
        seed=args.seed,
        epochs=1,
        lr=args.lr,
        experts=4,
        expert_k=2,
        text_moe=text_moe,
        feature_dim=128,
        hidden_dim=256,
        depth=4,
        n_heads=4,
        dataset_type="text",
        dataset_names=[args.data],
        batch_size=32,
        seq_len=64,
        max_chars=args.max_chars,
        max_steps_per_epoch=args.steps,
        max_eval_batches=args.eval_batches,
        save_dir=save_dir,
        model_type="moe",
    )


def build_model(cfg: dict, vocab_size: int) -> SimpleTransformer:
    return SimpleTransformer(
        vocab_size=vocab_size, d_model=cfg["feature_dim"], n_heads=cfg["n_heads"],
        n_layers=cfg["depth"], d_ff=cfg["hidden_dim"], max_len=cfg["seq_len"], dropout=0.1,
        num_experts=cfg["experts"] if cfg["text_moe"] else 0, expert_k=cfg["expert_k"],
    )


def run(args, name: str, text_moe: bool) -> dict:
    save_dir = os.path.join(args.work_dir, name)
    cfg = make_cfg(args, text_moe, save_dir)
    t0 = time.time()
    model, history = main(cfg)
    train_time = time.time() - t0
    steps = history["step_losses"]
    k = min(20, len(steps) // 2)

    # Checkpoint round trip into a freshly constructed model.
    meta_model_cfg = None
    ckpt_path = os.path.join(save_dir, "models", "last.pt")
    probe = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    meta_model_cfg = probe["cfg"]
    chars = probe["extra"]["chars"]
    fresh = build_model(meta_model_cfg, len(chars))
    load_checkpoint(ckpt_path, fresh)
    model.eval(); fresh.eval()
    tok = CharTokenizer("".join(chars))
    x = torch.tensor([tok.encode("ROMEO:\nWhat light through yonder window")], dtype=torch.long)
    with torch.no_grad():
        ckpt_diff = (model(x)[0] - fresh(x)[0]).abs().max().item()

    # Generation past the context window must keep every token.
    torch.manual_seed(args.seed)
    prompt = tok.encode("ROMEO:\n")
    n_new = cfg.seq_len + 36
    out = fresh.generate(prompt, max_new_tokens=n_new, temperature=0.8, top_k=20)
    sample = tok.decode(out.tolist())

    result = {
        "params": sum(p.numel() for p in model.parameters()),
        "steps": len(steps),
        "train_time_s": round(train_time, 1),
        "first_steps_loss": sum(steps[:k]) / k,
        "last_steps_loss": sum(steps[-k:]) / k,
        "val_loss": history["val_loss"][-1],
        "val_bpc": history["val_loss"][-1] / math.log(2),
        "val_acc": history["val_acc"][-1],
        "uniform_loss": math.log(len(chars)),
        "ckpt_max_logit_diff": ckpt_diff,
        "gen_len_ok": len(out) == len(prompt) + n_new,
        "sample": sample,
    }
    if text_moe:
        # Expert usage measured in eval mode on validation-style input.
        with torch.no_grad():
            _, aux, usage, *_ = fresh(x)
        per_layer = torch.stack([b.moe.expert_usage for b in fresh.blocks])
        ent = -(per_layer * (per_layer + 1e-9).log()).sum(-1) / math.log(per_layer.size(-1))
        result.update({
            "aux_loss": float(aux),
            "expert_usage_mean": [round(v, 3) for v in usage.tolist()],
            "max_expert_load": float(per_layer.max()),
            "min_routing_entropy": float(ent.min()),
        })
    return result


def checks(r: dict) -> dict:
    """Pass/fail criteria; thresholds are loose so the eval is stable across seeds."""
    c = {
        "loss_decreases": r["last_steps_loss"] < r["first_steps_loss"] - 0.5,
        "val_below_uniform": r["val_loss"] < r["uniform_loss"] - 0.5,
        "ckpt_roundtrip": r["ckpt_max_logit_diff"] < 1e-5,
        "generation_length": r["gen_len_ok"],
    }
    if "max_expert_load" in r:
        # With k=2 of 4 experts a balanced layer gives each expert 0.25 of the
        # assignments; > 0.5 would mean one expert takes every token's first slot.
        c["experts_balanced"] = r["max_expert_load"] < 0.5
    return c


def main_cli():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="input.txt")
    p.add_argument("--max-chars", type=int, default=700_000)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--eval-batches", type=int, default=20)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--work-dir", default="results/cpu_eval")
    p.add_argument("--out", default="results/cpu_eval/metrics.json")
    p.add_argument("--only", choices=["dense", "moe"], default=None)
    args = p.parse_args()
    torch.set_num_threads(max(1, os.cpu_count() or 1))

    results = {}
    for name, moe in (("dense", False), ("moe", True)):
        if args.only and args.only != name:
            continue
        r = run(args, name, moe)
        r["checks"] = checks(r)
        results[name] = r

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    for name, r in results.items():
        print(f"\n== {name} ({r['params']:,} params, {r['steps']} steps, {r['train_time_s']}s) ==")
        for key, v in r.items():
            if key not in ("sample", "checks", "params", "steps", "train_time_s"):
                print(f"  {key}: {v:.4f}" if isinstance(v, float) else f"  {key}: {v}")
        print("  checks:", r["checks"])
        print("  sample:", repr(r["sample"][:160]))
    ok = all(all(r["checks"].values()) for r in results.values())
    print("\nALL CHECKS PASSED" if ok else "\nSOME CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main_cli())
