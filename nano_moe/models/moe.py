import torch
import torch.nn as nn
import torch.nn.functional as F
from .attention import ContinuousAttention
from .reflection import ReflectiveAttentionBlock, ReflectiveAttentionCfg

class IntegratedMoE(nn.Module):
    """Image classifier: CNN encoder -> continuous attention -> reflective
    block -> mixture of classification-head experts chosen by a router.

    ``forward`` returns ``(logits, aux_loss, mean_gate)``, or with
    ``return_attention=True`` additionally ``(gate, expert_outs, think_stats)``.
    ``aux_loss`` is the Switch load-balancing loss (1.0 when balanced).
    """

    def __init__(self, num_experts, feature_dim, hidden_dim, num_classes,
                 router_hidden=64, k=1, gumbel=True, temperature=1.0):
        super().__init__()
        self.num_experts = num_experts
        self.k = k
        self.gumbel = gumbel
        self.temperature = temperature

        # 1. CNN Encoder (Spatial Features)
        self.cnn_encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, 1, 1), nn.ReLU(),
            nn.MaxPool2d(2)
        ) # Output: (B, 64, 7, 7)

        # 2. Continuous Attention Backbone (Sequence Modeling)
        # Treat 7x7 grid as sequence of 49 tokens
        self.seq_proj = nn.Linear(64, feature_dim)
        self.continuous_attn = ContinuousAttention(feature_dim, d_state=32)
        
        # 3. Reflective Reasoning (Thinking)
        self.reflective_cfg = ReflectiveAttentionCfg(
            dim=feature_dim,
            num_heads=4,
            max_iters=2, # Keep low for training speed
            use_energy_feedback=False # Simplified
        )
        self.reflective_block = ReflectiveAttentionBlock(self.reflective_cfg, block_size=49)

        # 4. MoE Router & Experts
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feature_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, num_classes)
            )
            for _ in range(num_experts)
        ])

        self.router = nn.Sequential(
            nn.Linear(feature_dim, router_hidden),
            nn.ReLU(),
            nn.Linear(router_hidden, num_experts)
        )

    def forward(self, x, dataset_idx=None, return_attention=False):
        B = x.shape[0]
        
        # 1. CNN Features
        cnn_feat = self.cnn_encoder(x) # (B, 64, 7, 7)
        
        # 2. Sequence Prep
        # Rearrange to (B, 49, 64)
        seq = cnn_feat.permute(0, 2, 3, 1).reshape(B, 49, 64)
        seq = self.seq_proj(seq) # (B, 49, feature_dim)
        
        # 3. Continuous Attention
        seq = seq + self.continuous_attn(seq)
        
        # 4. Reflective Thinking
        seq, think_stats = self.reflective_block(seq)
        
        # 5. Global Pooling for Classification
        global_feat = seq.mean(dim=1) # (B, feature_dim)
        
        # 6. MoE Routing
        logits_router = self.router(global_feat)
        # Load balancing is measured on the noise-free router distribution.
        clean_probs = F.softmax(logits_router, dim=1)

        if self.gumbel and self.training:
            gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits_router) + 1e-8) + 1e-8)
            logits_router = (logits_router + gumbel_noise) / self.temperature

        topk_vals, topk_idx = torch.topk(logits_router, self.k, dim=1)
        if self.k == 1:
            if self.gumbel and self.training:
                # Soft Gumbel-softmax sample: every expert runs during training.
                gate = F.softmax(logits_router, dim=1)
            else:
                gate = F.one_hot(topk_idx[:, 0], num_classes=self.num_experts).float()
        else:
            # Softmax renormalised over the selected experts so the router
            # receives gradient from the task loss (a constant 1/k gate does not).
            gate = torch.zeros_like(logits_router)
            gate.scatter_(1, topk_idx, F.softmax(topk_vals, dim=1))

        expert_outs = torch.stack([e(global_feat) for e in self.experts], dim=1)
        gate_expanded = gate.unsqueeze(-1)
        out = (expert_outs * gate_expanded).sum(dim=1)

        aux_loss = load_balancing_loss(clean_probs, topk_idx, self.num_experts)
        mean_gate = gate.mean(dim=0)

        if return_attention:
            return out, aux_loss, mean_gate, gate, expert_outs, think_stats

        return out, aux_loss, mean_gate


def load_balancing_loss(router_probs: torch.Tensor, expert_idx: torch.Tensor, num_experts: int) -> torch.Tensor:
    """Switch-Transformer load-balancing loss (Fedus et al., 2021, eq. 4).

    ``num_experts * sum_i f_i * P_i`` where ``f_i`` is the fraction of
    routing assignments sent to expert ``i`` and ``P_i`` the mean router
    probability for expert ``i``. It equals 1.0 when routing is uniform and
    grows towards ``num_experts`` as routing collapses onto one expert.

    Args:
        router_probs: (N, E) softmax over experts, computed without routing noise.
        expert_idx: (N, k) indices of the selected experts.
        num_experts: E.
    """
    counts = F.one_hot(expert_idx, num_experts).float().sum(dim=(0, 1))
    f = counts / counts.sum()
    p = router_probs.mean(dim=0)
    return num_experts * (f * p).sum()


class MoEFeedForward(nn.Module):
    """Token-level top-k mixture-of-experts feed-forward layer.

    Each token is routed to ``k`` of ``num_experts`` two-layer MLPs; the
    expert outputs are mixed with the router softmax renormalised over the
    selected experts, so the router receives gradient from the LM loss.

    After each forward pass ``self.aux_loss`` holds the load-balancing loss
    (see :func:`load_balancing_loss`) and ``self.expert_usage`` the fraction
    of assignments per expert (detached), for logging.
    """

    def __init__(self, d_model: int, d_ff: int, num_experts: int = 4, k: int = 2, dropout: float = 0.0):
        super().__init__()
        assert 1 <= k <= num_experts
        self.num_experts = num_experts
        self.k = k
        self.router = nn.Linear(d_model, num_experts, bias=False)
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model), nn.Dropout(dropout))
            for _ in range(num_experts)
        ])
        self.aux_loss = torch.tensor(0.0)
        self.expert_usage = torch.zeros(num_experts)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        flat = x.reshape(-1, shape[-1])  # (N, D)
        probs = F.softmax(self.router(flat), dim=-1)  # (N, E)
        topk_p, topk_idx = probs.topk(self.k, dim=-1)  # (N, k)
        weights = topk_p / topk_p.sum(dim=-1, keepdim=True)

        out = torch.zeros_like(flat)
        for e, expert in enumerate(self.experts):
            token_idx, slot = (topk_idx == e).nonzero(as_tuple=True)
            if token_idx.numel() == 0:
                continue
            y = expert(flat[token_idx]) * weights[token_idx, slot].unsqueeze(-1)
            out.index_add_(0, token_idx, y)

        self.aux_loss = load_balancing_loss(probs, topk_idx, self.num_experts)
        with torch.no_grad():
            counts = F.one_hot(topk_idx, self.num_experts).float().sum(dim=(0, 1))
            self.expert_usage = counts / counts.sum()
        return out.reshape(shape)
