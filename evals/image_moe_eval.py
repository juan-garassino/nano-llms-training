"""Short CPU check that ``IntegratedMoE`` learns and routes on synthetic images.

Ten random 28x28 templates plus Gaussian noise (sigma 0.5) form a 10-class
task a linear model solves easily, so chance-level accuracy (0.10) after
training points at a broken model rather than a hard task. Reports eval
accuracy and the router's top-k assignment balance for k=1 and k=2 over two
seeds. Takes about a minute on 4 CPU cores.

    python evals/image_moe_eval.py [--steps 150] [--src PATH_TO_OTHER_CHECKOUT]
"""

import argparse
import json
import math
import os
import sys

import torch
import torch.nn.functional as F


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=150)
    p.add_argument("--src", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   help="repo root to import nano_moe from (lets you compare checkouts)")
    p.add_argument("--out", default=None)
    args = p.parse_args()
    sys.path.insert(0, args.src)
    from nano_moe.models.moe import IntegratedMoE

    results = []
    for seed in (0, 1):
        for k in (1, 2):
            torch.manual_seed(seed)
            model = IntegratedMoE(num_experts=4, feature_dim=32, hidden_dim=64, num_classes=10, k=k)
            opt = torch.optim.Adam(model.parameters(), 1e-3)
            templates = torch.randn(10, 1, 28, 28)
            y = torch.randint(0, 10, (512,))
            x = templates[y] + 0.5 * torch.randn(512, 1, 28, 28)
            for _ in range(args.steps):
                i = torch.randint(0, 512, (32,))
                out, aux, _ = model(x[i])
                loss = F.cross_entropy(out, y[i]) + 0.01 * aux
                opt.zero_grad()
                loss.backward()
                opt.step()

            model.eval()
            captured = {}
            # Capture router logits so the same probe works on any checkout.
            hook = model.router.register_forward_hook(lambda m, i, o: captured.update(logits=o))
            with torch.no_grad():
                out = model(x[:256])[0]
            hook.remove()
            acc = (out.argmax(1) == y[:256]).float().mean().item()
            idx = captured["logits"].topk(k, dim=1).indices
            load = F.one_hot(idx, 4).float().sum((0, 1)) / idx.numel()
            entropy = -(load * (load + 1e-9).log()).sum().item() / math.log(4)
            results.append({"seed": seed, "k": k, "acc": round(acc, 3),
                            "max_expert_load": round(load.max().item(), 3),
                            "routing_entropy": round(entropy, 3)})
            print(results[-1])

    mean_acc = sum(r["acc"] for r in results) / len(results)
    print(f"mean acc {mean_acc:.3f} (chance 0.100)")
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"runs": results, "mean_acc": mean_acc}, f, indent=2)


if __name__ == "__main__":
    main()
