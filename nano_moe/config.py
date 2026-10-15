from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class TrainingConfig:
    seed: int = 42
    epochs: int = 3
    lr: float = 1e-3
    experts: int = 4
    feature_dim: int = 128
    hidden_dim: int = 256
    # Data
    dataset_type: str = "text" # "image" or "text" or "arc"
    # ["mnist", "fashionmnist"], ["wikitext"], or a path to a local text file
    # (e.g. ["input.txt"]), which is loaded character-level without downloads.
    dataset_names: list = field(default_factory=lambda: ["tinyshakespeare"])
    batch_size: int = 32 # Reduced for text sequence length
    seq_len: int = 128 # For text
    save_dir: str = "results/shakespeare"
    num_workers: int = 0
    
    # Short-run controls (None = full epoch / full validation set)
    max_steps_per_epoch: Optional[int] = None
    max_eval_batches: Optional[int] = None
    max_chars: Optional[int] = None # Truncate a local text file (char-level data only)
    
    # Model
    model_type: str = "moe" # "moe", "phase", "phase_symbolic", "true_phase", "universal_phase", "quantum_diffusion"
    
    # Text MoE: when True, model_type="moe" on text uses top-k MoE feed-forward
    # layers with `experts` experts; False keeps the dense SimpleTransformer.
    text_moe: bool = False
    expert_k: int = 2
    aux_weight: float = 0.01 # Weight of the load-balancing loss
    
    # SFPT Params
    n_freqs: int = 2048
    top_k: int = 32
    expansion: int = 4
    depth: int = 4
    n_heads: int = 8
    
    # Phase-Symbolic Params
    n_ops: int = 16
    max_program_len: int = 5
