import torch
import torch.nn as nn
import numpy as np
import os
import random
from typing import Optional
from rich.console import Console
from rich.panel import Panel

from .config import TrainingConfig
from .datasets.loaders import get_multimnist_loaders
from .datasets.text import get_text_loaders, get_char_loaders
from .models.moe import IntegratedMoE
from .models.phase import SparseFourierPhaseTransformer
from .models.phase_symbolic import HybridPhaseSymbolicARC
from .models.true_phase import TrueHolographicSFPT
from .models.universal_phase import UniversalSFPT
from .models.quantum_diffusion import QuantumDenoiser
from .models.transformer import SimpleTransformer
from .training.tracker import ExperimentTracker
from .training.trainer import train_epoch, eval_model
from .training.checkpoint import save_checkpoint

console = Console()

def main(cfg: Optional[TrainingConfig] = None):
    """Train the model selected by ``cfg`` and return ``(model, history)``.

    ``history`` holds per-epoch train loss, val loss and val accuracy. The
    final weights are written to ``{cfg.save_dir}/models/last.pt``.
    Specialised model types (``phase_symbolic``, ``quantum_diffusion``) are
    dispatched to their own training loops and return ``None``.
    """
    # Configuration
    if cfg is None:
        cfg = TrainingConfig()
    
    # Setup
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    console.print(Panel(f"🚀 Starting Pipeline on {device} | Model: {cfg.model_type} | Data: {cfg.dataset_type}", style="bold green"))
    
    # Data
    tokenizer = None
    if cfg.dataset_type == "text":
        # Use first dataset name as the target text dataset
        if not cfg.dataset_names:
            cfg.dataset_names = ["tinyshakespeare"]
        dataset_name = cfg.dataset_names[0]
        if os.path.isfile(dataset_name):
            loaders, vocab_size, tokenizer = get_char_loaders(
                dataset_name, cfg.batch_size, cfg.seq_len, num_workers=cfg.num_workers,
                max_chars=cfg.max_chars, seed=cfg.seed)
            cfg.dataset_names = [dataset_name]
        else:
            loaders, vocab_size = get_text_loaders(dataset_name, cfg.batch_size, cfg.seq_len, num_workers=cfg.num_workers)
        num_classes = vocab_size
        # Override feature_dim if needed? No, feature_dim is model dim.
    else:
        if not cfg.dataset_names:
            cfg.dataset_names = ["mnist", "fashionmnist"]
        loaders, class_counts = get_multimnist_loaders(cfg.dataset_names, cfg.batch_size, num_workers=cfg.num_workers)
        num_classes = max(class_counts.values())
    
    # Model
    if cfg.model_type == "sfpt":
        model = SparseFourierPhaseTransformer(
            vocab_size=num_classes,
            dim=cfg.feature_dim,
            depth=cfg.depth,
            n_heads=cfg.n_heads,
            n_freqs=cfg.n_freqs,
            top_k=cfg.top_k,
            expansion=cfg.expansion
        ).to(device)
    elif cfg.model_type == "phase_symbolic":
        model = HybridPhaseSymbolicARC(
            vocab_size=num_classes,
            dim=cfg.feature_dim,
            n_layer=cfg.depth,
            n_heads=cfg.n_heads,
            n_freqs=cfg.n_freqs,
            top_k=cfg.top_k,
            expansion=cfg.expansion,
            dropout=0.1,
            n_ops=cfg.n_ops,
            max_program_len=cfg.max_program_len
        ).to(device)
    elif cfg.model_type == "true_phase":
        model = TrueHolographicSFPT(
            vocab_size=num_classes,
            layers=cfg.depth,
            dim=cfg.feature_dim
        ).to(device)
    elif cfg.model_type == "universal_phase":
        model = UniversalSFPT(
            vocab_size=num_classes,
            layers=cfg.depth,
            dim=cfg.feature_dim,
            freqs=cfg.n_freqs,
            A=3.0
        ).to(device)
    elif cfg.model_type == "quantum_diffusion":
        # Quantum Diffusion is specialized for ARC, so we might need to handle it differently
        # For now, just instantiate it.
        model = QuantumDenoiser(dim=cfg.feature_dim).to(device)
    elif cfg.model_type == "moe": # Assuming "moe" is the type for IntegratedMoE
        if cfg.dataset_type == "text":
            # IntegratedMoE is an image classifier; for text use the GPT-style
            # transformer, with MoE feed-forward layers only if cfg.text_moe.
            model = SimpleTransformer(
                vocab_size=num_classes,
                d_model=cfg.feature_dim,
                n_heads=cfg.n_heads,
                n_layers=cfg.depth,
                d_ff=cfg.hidden_dim,
                max_len=cfg.seq_len,
                dropout=0.1,
                num_experts=cfg.experts if cfg.text_moe else 0,
                expert_k=cfg.expert_k,
            ).to(device)
        else:
            model = IntegratedMoE(
                num_experts=cfg.experts,
                feature_dim=cfg.feature_dim,
                hidden_dim=cfg.hidden_dim,
                num_classes=num_classes
            ).to(device)
    else:
        raise ValueError(f"Unknown model type: {cfg.model_type}")

    # Dispatch to specialized training loops if needed
    if cfg.model_type == "phase_symbolic" or cfg.dataset_type == "arc":
        from .train_arc import train_arc
        # Pass config to train_arc if it accepts it, or just run it
        # Currently train_arc() creates its own config, we should update it to accept cfg
        # For now, let's just call it.
        train_arc(cfg) 
        return

    if cfg.model_type == "quantum_diffusion":
        from .models.quantum_diffusion import train_quantum_diffusion
        train_quantum_diffusion(cfg)
        return
    
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    criterion = nn.CrossEntropyLoss()
    tracker = ExperimentTracker(cfg.save_dir)
    history = {"train_loss": [], "val_loss": [], "val_acc": [], "step_losses": []}
    
    # Training Loop
    for epoch in range(1, cfg.epochs + 1):
        losses, gates, thinking = train_epoch(model, loaders, optimizer, criterion, device, cfg.dataset_names, tracker, epoch,
                                              max_steps=cfg.max_steps_per_epoch, aux_weight=cfg.aux_weight,
                                              grad_clip=cfg.grad_clip)
        
        val_losses = {}
        accs = eval_model(model, loaders, device, cfg.dataset_names,
                          max_batches=cfg.max_eval_batches, losses=val_losses)
        avg_loss = np.mean(losses)
        history["train_loss"].append(float(avg_loss))
        history["val_loss"].append(float(np.mean(list(val_losses.values()))))
        history["val_acc"].append(float(np.mean(list(accs.values()))))
        history["step_losses"].extend(float(l) for l in losses)
        avg_think = np.mean(thinking) if thinking else 0.0
        
        # Log
        tracker.log_epoch_metrics(epoch, avg_loss, np.mean(list(accs.values())), accs, torch.stack(gates).mean(0), thinking_depth=avg_think)
        
        # Display
        console.print(f"[bold]Epoch {epoch}[/]: Loss={avg_loss:.4f}, Val Loss={history['val_loss'][-1]:.4f}, "
                      f"Avg Acc={np.mean(list(accs.values())):.4f}, Thinking Depth={avg_think:.2f}")
        
    tracker.save_metrics()
    extra = {"history": history}
    if tokenizer is not None:
        extra["chars"] = tokenizer.chars
    save_checkpoint(os.path.join(cfg.save_dir, "models", "last.pt"), model, optimizer, cfg, **extra)
    console.print(f"[bold green]Training Complete! Results saved to {cfg.save_dir}[/]")
    return model, history

if __name__ == "__main__":
    main()
