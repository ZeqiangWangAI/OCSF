"""End-to-End Token-Level Bidirectional Diffusion OCSF (OCSF-E2E).

Replaces coarse 256-word frozen chunking with single-pass token-level bidirectional attention
(up to 4,096 tokens) + multilingual-preserving upper-layer LoRA + Token-to-Channel Clinical
Attention (the exact ontology-invariant generalization of PLM-ICD's LAAT decoder).

Key properties:
  1. Zero hardcoded code-ID parameters: all decoder layers operate strictly on R ontology-invariant
     clinical channels in R^{R x K}, read out by frozen Self-Normalized Tikhonov Deconvolution W_{B,r}.
  2. Multilingual preservation: token embeddings and bottom half of Transformer layers stay frozen
     so Spanish/Greek/Chinese inputs enter the shared semantic manifold before upper-layer clinical
     reasoning (LoRA on top half of layers).
  3. Streaming GPU deconvolution scoring: scores arbitrary target codebooks (up to 185,458 codes)
     batch-by-batch directly on GPU with O(1) host RAM even at R=512 and K=2560.
"""

from __future__ import annotations

import contextlib
import math
from typing import Iterable
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from .state_matrix import DeconvolutionOperator


class LoRALinear(nn.Module):
    """Lightweight native LoRA wrapper around a frozen nn.Linear (no external PEFT quirks)."""

    def __init__(self, base: nn.Linear, r: int = 32, alpha: float = 64.0, dropout: float = 0.05) -> None:
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        self.r = int(r)
        self.scale = float(alpha) / float(r)
        self.drop = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        dtype = base.weight.dtype
        device = base.weight.device
        self.lora_a = nn.Parameter(torch.empty((r, base.in_features), device=device, dtype=torch.float32))
        self.lora_b = nn.Parameter(torch.zeros((base.out_features, r), device=device, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        self._base_dtype = dtype

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        x_d = self.drop(x)
        delta = F.linear(F.linear(x_d, self.lora_a.to(x_d.dtype)), self.lora_b.to(x_d.dtype)) * self.scale
        return out + delta.to(out.dtype)


def inject_upper_layer_lora(
    backbone: nn.Module,
    r: int = 32,
    alpha: float = 64.0,
    dropout: float = 0.05,
    freeze_ratio: float = 0.5,
    target_suffixes: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"),
) -> tuple[int, int]:
    """Freeze entire backbone, then inject LoRA into top (1 - freeze_ratio) transformer layers.

    Returns (n_frozen_layers, n_lora_modules).
    """
    for p in backbone.parameters():
        p.requires_grad = False

    # Locate Transformer layer list on Qwen3 / PPLXQwen3 / generic HF models
    layers = None
    for attr_path in (("layers",), ("model", "layers"), ("encoder", "layer")):
        obj = backbone
        for a in attr_path:
            obj = getattr(obj, a, None)
            if obj is None:
                break
        if obj is not None and isinstance(obj, nn.ModuleList):
            layers = obj
            break
    if layers is None:
        raise ValueError(f"Could not locate transformer ModuleList on {type(backbone).__name__}")

    n_layers = len(layers)
    start_layer = int(round(n_layers * freeze_ratio))
    n_injected = 0

    for layer_idx in range(start_layer, n_layers):
        layer = layers[layer_idx]
        for mod_name, mod in list(layer.named_modules()):
            if not mod_name:
                continue
            leaf = mod_name.split(".")[-1]
            if leaf in target_suffixes and isinstance(mod, nn.Linear):
                parts = mod_name.split(".")
                parent = layer
                for p_name in parts[:-1]:
                    parent = getattr(parent, p_name)
                setattr(parent, parts[-1], LoRALinear(mod, r=r, alpha=alpha, dropout=dropout))
                n_injected += 1

    return start_layer, n_injected


class TokenChannelStateDecoder(nn.Module):
    """Token-Level Channel Attention Decoder mapping token hidden states H (B, L, K) -> U (B, R, K).

    Exact ontology-invariant generalization of PLM-ICD's LAAT decoder:
      - Head 0: Frozen-space multilingual token-to-channel cosine routing (zero W_V/W_O rotation).
      - Multi-Head SDPA: Learned clinical detail attention directly over all L <= 4096 tokens,
        anchored on semantic channel centroids A_c (FlashAttention-compatible).
      - Cross-Channel Transformer: Models clinical co-occurrence, negation, and coding rules across
        all R channels, outputting the dual-path superposition state matrix U(x) in R^{R x K}.
    """

    def __init__(
        self,
        anchors: torch.Tensor,  # (R, K) unit-norm channel centroids fit strictly on source Z_A
        n_heads: int = 8,
        n_channel_layers: int = 2,
        dropout: float = 0.1,
        route_temp: float = 0.07,
    ) -> None:
        super().__init__()
        n_channels, dim = anchors.shape
        if dim % n_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by n_heads={n_heads}")
        self.n_channels = n_channels
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.route_temp = float(route_temp)

        self.register_buffer("anchors", F.normalize(anchors.float(), p=2, dim=-1))

        # Token-level clinical salience scorer (zero-init -> pure cosine routing at step 0)
        self.tok_salience = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim // 4),
            nn.GELU(),
            nn.Linear(dim // 4, 1),
        )
        nn.init.zeros_(self.tok_salience[-1].weight)
        nn.init.zeros_(self.tok_salience[-1].bias)

        # Learned token-to-channel multi-head FlashAttention (LAAT analogue, ontology-invariant!)
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)
        self.tok_norm = nn.LayerNorm(dim)
        self.q_norm = nn.LayerNorm(dim)
        self.cross_gate = nn.Parameter(torch.tensor(0.25))

        # Cross-channel clinical Transformer over R channels
        self.sim_proj = nn.Linear(2, dim)
        self.channel_in_norm = nn.LayerNorm(dim)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=n_heads,
            dim_feedforward=dim * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.channel_encoder = nn.TransformerEncoder(
            enc_layer, num_layers=n_channel_layers, enable_nested_tensor=False
        )

        # Dual-path state readout: anchored evidence + zero-init multi-code superposition residual
        self.state_norm = nn.LayerNorm(dim)
        self.vec_head = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
        )
        self.gate_delta = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim // 4),
            nn.GELU(),
            nn.Linear(dim // 4, 1),
        )
        self.peak_scale = nn.Parameter(torch.tensor(4.0))
        self.peak_shift = nn.Parameter(torch.tensor(0.20))

        nn.init.zeros_(self.vec_head[-1].weight)
        nn.init.zeros_(self.vec_head[-1].bias)
        nn.init.zeros_(self.gate_delta[-1].weight)
        nn.init.zeros_(self.gate_delta[-1].bias)
        self._vec_scale = 1024.0 / float(dim)  # exact muP width invariance (1.0 at d=1024, 0.4 at d=2560)
        self._last_delta: torch.Tensor | None = None

    def forward(
        self,
        token_states: torch.Tensor,  # (B, L, K) L2-normalized token vectors
        attention_mask: torch.Tensor,  # (B, L) 1/True for valid tokens
        anchors_override: torch.Tensor | None = None,
    ) -> torch.Tensor:
        bsz, seq_len, dim = token_states.shape
        valid = attention_mask.bool()
        anchors = (self.anchors if anchors_override is None else anchors_override).to(token_states.dtype)

        # 1. Frozen-space multilingual token routing (zero W_V/W_O rotation!)
        salience = self.tok_salience(token_states).squeeze(-1)  # (B, L)
        sim = torch.einsum("rk,blk->brl", anchors, token_states)  # (B, R, L)
        route_logits = (sim + salience.unsqueeze(1)) / self.route_temp
        route_logits = route_logits.masked_fill(~valid.unsqueeze(1), -1e4)
        attn0 = F.softmax(route_logits.float(), dim=-1).to(token_states.dtype)
        evidence = torch.einsum("brl,blk->brk", attn0, token_states)  # (B, R, K)
        evidence = F.normalize(evidence.float(), p=2, dim=-1).to(token_states.dtype)

        # Language-invariant peak & mean token cosine statistics per channel
        sim_masked = sim.masked_fill(~valid.unsqueeze(1), -1.0)
        peak_sim = sim_masked.max(dim=-1, keepdim=True).values  # (B, R, 1)
        denom = valid.float().sum(dim=-1, keepdim=True).clamp(min=1.0).unsqueeze(1)
        mean_sim = (sim.float() * valid.float().unsqueeze(1)).sum(dim=-1, keepdim=True) / denom
        sim_stats = torch.cat([peak_sim.float(), mean_sim], dim=-1).to(token_states.dtype)

        # 2. Learned Multi-Head Token-to-Channel Attention via SDPA (FlashAttention)
        q_in = self.q_norm(evidence + anchors.unsqueeze(0))
        kv_in = self.tok_norm(token_states)
        q = self.q_proj(q_in).view(bsz, self.n_channels, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(kv_in).view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(kv_in).view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        # Boolean SDPA mask: True means attend, shape (B, 1, 1, L) broadcasts cleanly over heads and R
        sdpa_mask = valid[:, None, None, :]
        # Hopper cuDNN fused SDPA backward crashes when query length R < 8 (e.g., R=1 ablation)
        if self.n_channels < 8:
            with sdpa_kernel(SDPBackend.MATH):
                cross_ctx = F.scaled_dot_product_attention(q, k, v, attn_mask=sdpa_mask)
        else:
            cross_ctx = F.scaled_dot_product_attention(q, k, v, attn_mask=sdpa_mask)
        cross_ctx = cross_ctx.transpose(1, 2).contiguous().view(bsz, self.n_channels, dim)
        cross_ctx = self.o_proj(cross_ctx)

        # 3. Cross-channel clinical Transformer + dual-path superposition readout
        c = self.channel_in_norm(evidence + self.cross_gate * cross_ctx + self.sim_proj(sim_stats))
        if self.n_channels < 8:
            with sdpa_kernel(SDPBackend.MATH):
                c = self.channel_encoder(c)
        else:
            c = self.channel_encoder(c)

        delta = self.vec_head(self.state_norm(c)) * self._vec_scale
        self._last_delta = delta
        gate = F.softplus(self.peak_scale * (peak_sim - self.peak_shift) + self.gate_delta(c))
        return (gate * evidence + delta).float()


class ClosedVocabLAATHead(nn.Module):
    """In-domain closed-vocabulary LAAT control head (PLM-ICD equivalent on exact same backbone)."""

    def __init__(self, dim: int, n_codes: int, proj_dim: int = 512, dropout: float = 0.2) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, proj_dim)
        self.label_queries = nn.Linear(proj_dim, n_codes, bias=False)
        self.out_weight = nn.Parameter(torch.empty((n_codes, dim)))
        # Prior-logit initialization log(p/(1-p)) prevents massive negative-class gradient shock at step 0
        self.out_bias = nn.Parameter(torch.full((n_codes,), -6.5))
        nn.init.normal_(self.label_queries.weight, std=0.02)
        nn.init.normal_(self.out_weight, std=0.02)

    def forward(self, token_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        h = self.drop(self.norm(token_states))  # (B, L, K)
        z = torch.tanh(self.proj(h))  # (B, L, d_p)
        attn_logits = self.label_queries(z).transpose(1, 2)  # (B, C, L)
        attn_logits = attn_logits.masked_fill(~attention_mask.bool().unsqueeze(1), -1e4)
        attn = F.softmax(attn_logits.float(), dim=-1).to(h.dtype)
        # Label-specific representations V = A H (B, C, K) -> elementwise dot with out_weight
        v = torch.bmm(attn, h)  # (B, C, K)
        return (v * self.out_weight.to(h.dtype).unsqueeze(0)).sum(dim=-1).float() + self.out_bias.float()


class TokenLevelOCSFModel(nn.Module):
    """Complete End-to-End Token-Level Bidirectional Diffusion OCSF wrapper."""

    def __init__(
        self,
        backbone: nn.Module,
        decoder: TokenChannelStateDecoder,
        apply_tanh: bool = True,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.decoder = decoder
        self.apply_tanh = apply_tanh

    def encode_tokens(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        h = out.last_hidden_state
        if self.apply_tanh:
            h = torch.tanh(h)
        h = F.normalize(h.float(), p=2, dim=-1).to(out.last_hidden_state.dtype)
        return h * attention_mask.unsqueeze(-1).to(h.dtype)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        anchors_override: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h = self.encode_tokens(input_ids, attention_mask)
        return self.decoder(h, attention_mask, anchors_override=anchors_override)


class StreamingGPUDeconvScorer:
    """Fast GPU-resident batch scorer for any compiled DeconvolutionOperator (up to 185,458 codes).

    Avoids materializing (N, R, K) in host RAM: streams (B, R, K) -> (B, |B|) in <0.1 ms/batch.
    """

    def __init__(self, op: DeconvolutionOperator, device: torch.device, dtype: torch.dtype = torch.float16) -> None:
        self.op = op
        self.device = device
        self.dtype = dtype
        self.n_codes = op.n_codes
        self.n_channels = op.n_channels
        # Keep non-empty channels on GPU; for huge codebooks (CodiEsp 185k * 1024 * 2B = 380 MB), fits trivially in GH200 120 GB!
        self._gpu_items: list[tuple[int, torch.Tensor, torch.Tensor]] = []
        for r, (idx, w) in enumerate(zip(op.chan_indices, op.chan_weights)):
            if len(idx) == 0:
                continue
            self._gpu_items.append(
                (r, torch.from_numpy(idx.astype(np.int64)).to(device), torch.from_numpy(w).to(device=device, dtype=dtype))
            )

    @torch.inference_mode()
    def score_tensor(self, u_batch: torch.Tensor) -> np.ndarray:
        """Score a batch of 2D state matrices u_batch (B, R, K) -> numpy float32 (B, |B|)."""
        u_d = u_batch.to(device=self.device, dtype=self.dtype)
        out = torch.empty((u_d.shape[0], self.n_codes), device=self.device, dtype=torch.float32)
        for r, idx_t, w_t in self._gpu_items:
            out[:, idx_t] = (u_d[:, r, :] @ w_t.T).float()
        return out.cpu().numpy()


def make_length_bucketed_batches(
    lengths: Iterable[int],
    max_tokens: int,
    max_batch_size: int = 32,
    rng: np.random.Generator | None = None,
    bucket_noise: float = 0.08,
) -> list[list[int]]:
    """Group sequence indices into near-uniform length batches minimizing padding waste (<5%)."""
    lens = np.asarray(list(lengths), dtype=np.int64)
    n = len(lens)
    if n == 0:
        return []
    if rng is not None and bucket_noise > 0.0:
        keys = lens * (1.0 + rng.uniform(-bucket_noise, bucket_noise, size=n))
    else:
        keys = lens.astype(np.float64)
    order = np.argsort(keys)

    batches: list[list[int]] = []
    cur: list[int] = []
    cur_max = 0
    for idx in order.tolist():
        seq_len = int(lens[idx])
        next_max = max(cur_max, seq_len)
        if cur and (next_max * (len(cur) + 1) > max_tokens or len(cur) >= max_batch_size):
            batches.append(cur)
            cur = [idx]
            cur_max = seq_len
        else:
            cur.append(idx)
            cur_max = next_max
    if cur:
        batches.append(cur)

    if rng is not None:
        perm = rng.permutation(len(batches))
        batches = [batches[i] for i in perm.tolist()]
    return batches
