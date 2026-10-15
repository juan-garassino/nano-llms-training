import inspect

import torch
import numpy as np
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.console import Console

try:
    console = Console()
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

def _forward_kwargs(model, y):
    """Pass ``classification=False`` to models that accept it when targets are sequences."""
    if 'classification' in inspect.signature(model.forward).parameters and y.dim() == 2:
        return {'classification': False}
    return {}


def train_epoch(model, loaders, optimizer, criterion, device, dataset_names, tracker, epoch,
                max_steps=None, aux_weight=0.01):
    """Run one epoch, stepping through all datasets in lock-step.

    Args:
        max_steps: Optional cap on optimizer steps (used for short CPU runs).
        aux_weight: Weight of the model's auxiliary (e.g. MoE load-balancing) loss.

    Returns:
        (losses, gates, thinking): per-step total loss, per-step mean gate
        vector (``torch.zeros(1)`` for models without a router) and per-step
        reflective-thinking depth.
    """
    model.train()
    steps = min(len(loaders[name]["train_loader"]) for name in dataset_names)
    if max_steps is not None:
        steps = min(steps, max_steps)
    iters = {name: iter(loaders[name]["train_loader"]) for name in dataset_names}
    
    all_losses = []
    all_gates = []
    all_thinking = []

    progress_ctx = Progress(
        SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
        BarColumn(), TaskProgressColumn(), console=console
    ) if RICH_AVAILABLE else None

    if progress_ctx: progress_ctx.start()
    task = progress_ctx.add_task("Training...", total=steps) if progress_ctx else None

    for step in range(steps):
        optimizer.zero_grad()
        total_loss = 0.0
        batch_gates = []
        batch_think = []

        for idx, name in enumerate(dataset_names):
            try:
                x, y = next(iters[name])
            except StopIteration:
                iters[name] = iter(loaders[name]["train_loader"])
                x, y = next(iters[name])

            x, y = x.to(device), y.to(device)
            
            kwargs = {'return_attention': True, **_forward_kwargs(model, y)}
            logits, aux, gate, _, _, think_stats = model(x, idx, **kwargs)
            
            if logits.dim() == 3: # (B, L, V)
                # Flatten for CrossEntropyLoss
                B, L, V = logits.shape
                loss = criterion(logits.reshape(-1, V), y.reshape(-1)) + aux_weight * aux
            else:
                loss = criterion(logits, y) + aux_weight * aux
                
            total_loss += loss
            
            if gate is not None:
                batch_gates.append(gate.detach().cpu())
            
            # Track thinking depth
            if think_stats and 'iterations_used' in think_stats:
                batch_think.append(think_stats['iterations_used'].mean().item())

        total_loss.backward()
        optimizer.step()
        
        all_losses.append(total_loss.item())
        if batch_gates:
            all_gates.append(torch.stack(batch_gates).mean(0))
        else:
            all_gates.append(torch.zeros(1)) # Dummy
            
        if batch_think:
            all_thinking.append(np.mean(batch_think))
        
        if progress_ctx: progress_ctx.update(task, advance=1)

    if progress_ctx: progress_ctx.stop()
    
    return all_losses, all_gates, all_thinking

@torch.no_grad()
def eval_model(model, loaders, device, dataset_names, max_batches=None, losses=None):
    """Accuracy per dataset on its ``test_loader`` (token accuracy for sequence targets).

    Args:
        max_batches: Optional cap on evaluated batches per dataset.
        losses: Optional dict; if given it is filled with the mean
            cross-entropy per dataset.
    """
    model.eval()
    accs = {}
    for name in dataset_names:
        loader = loaders[name]["test_loader"]
        correct, total = 0, 0
        loss_sum, loss_n = 0.0, 0
        for b, (x, y) in enumerate(loader):
            if max_batches is not None and b >= max_batches:
                break
            x, y = x.to(device), y.to(device)
            
            # Models return 3- or 6-tuples; logits are always first.
            logits = model(x, **_forward_kwargs(model, y))[0]
            loss_sum += torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction='sum').item()
            loss_n += y.numel()
            
            if logits.dim() == 3: # Sequence
                pred = logits.argmax(2) # (B, L)
                correct += (pred==y).sum().item()
                total += y.numel()
            else:
                pred = logits.argmax(1)
                correct += (pred==y).sum().item()
                total += y.size(0)
        accs[name] = correct/total
        if losses is not None:
            losses[name] = loss_sum / loss_n
    return accs
