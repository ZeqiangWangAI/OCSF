"""PyTorch architectures for OCSF and controlled 1D baselines (B1/B2).

All OCSF parameters are 100% ontology-invariant: no layer dimension depends on |A| or |B|.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def zlpr_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Zero-bounded multi-label pairwise rank loss (Su et al., 2022).

    Prevents extreme-sparsity all-zero collapse without brittle pos_weight tuning and focuses
    gradients directly on hard sibling false-positives vs true positives.
    """
    y_pos = targets.bool()
    neg_logits = logits.masked_fill(y_pos, -1e4)
    pos_logits = (-logits).masked_fill(~y_pos, -1e4)
    zeros = torch.zeros((logits.shape[0], 1), device=logits.device, dtype=logits.dtype)
    loss_neg = torch.logsumexp(torch.cat([neg_logits, zeros], dim=1), dim=1)
    loss_pos = torch.logsumexp(torch.cat([pos_logits, zeros], dim=1), dim=1)
    return (loss_neg + loss_pos).mean()


class Flat1DClassifier(nn.Module):
    """B1_CAST_1D Baseline: bias-free cosine 1D classifier W_A initialized from Z_A (CAST-projected to any B)."""

    def __init__(
        self,
        dim: int,
        n_source_codes: int,
        z_source_init: torch.Tensor | None = None,
        dropout: float = 0.1,
        logit_scale: float = 16.0,
        logit_margin: float = 0.40,
    ) -> None:
        super().__init__()
        self.logit_scale = float(logit_scale)
        self.logit_margin = float(logit_margin)
        self.attn_score = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 1))
        nn.init.zeros_(self.attn_score[-1].weight)
        nn.init.zeros_(self.attn_score[-1].bias)
        self.proj = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Dropout(dropout),
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
        )
        nn.init.zeros_(self.proj[-1].weight)
        nn.init.zeros_(self.proj[-1].bias)
        self.classifier = nn.Linear(dim, n_source_codes, bias=False)
        if z_source_init is not None:
            with torch.no_grad():
                self.classifier.weight.copy_(F.normalize(z_source_init.float(), p=2, dim=-1))

    def encode_doc(self, chunks: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        valid = mask.bool()
        attn_logits = self.attn_score(chunks).squeeze(-1).masked_fill(~valid, -1e4)
        attn = F.softmax(attn_logits, dim=-1).unsqueeze(-1)
        pooled = F.normalize((chunks * attn).sum(dim=1), p=2, dim=-1)
        return F.normalize(pooled + self.proj(pooled), p=2, dim=-1)

    def normalized_weights(self) -> torch.Tensor:
        return F.normalize(self.classifier.weight, p=2, dim=-1)

    def forward(self, chunks: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        doc_vec = self.encode_doc(chunks, mask)
        cos_sim = F.linear(doc_vec, self.normalized_weights())
        logits = self.logit_scale * (cos_sim - self.logit_margin)
        return logits, doc_vec


class Flat1DStatePredictor(nn.Module):
    """B1_1D_State Baseline: exact R=1 1D state vector special case of OCSF (zero code-ID params)."""

    def __init__(self, dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.attn_score = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 1))
        nn.init.zeros_(self.attn_score[-1].weight)
        nn.init.zeros_(self.attn_score[-1].bias)
        self.mlp = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Dropout(dropout),
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)
        self.gain = nn.Parameter(torch.tensor(1.5))

    def forward(self, chunks: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        valid = mask.bool()
        attn_logits = self.attn_score(chunks).squeeze(-1).masked_fill(~valid, -1e4)
        attn = F.softmax(attn_logits, dim=-1).unsqueeze(-1)
        pooled = F.normalize((chunks * attn).sum(dim=1), p=2, dim=-1)
        return self.gain * pooled + self.mlp(pooled)


class StateFieldPredictor(nn.Module):
    """Step 1/2: Dual-Path Ontology-Invariant 2D Clinical State Matrix U(x) in R^{R x K}.

    Combines:
      1. Frozen-space multilingual chunk routing (zero W_V/W_O rotation) -> multilingual anchor state U_anchor.
      2. Multi-head channel-to-chunk cross-attention + Cross-Channel Transformer -> unconstrained
         multi-code superposition residual Delta_U(x) (zero-init, so step 0 starts at exact zero-shot
         anchor and training builds crisp multi-code vector superpositions sum_{i in y_r} z_i).
    """

    def __init__(
        self,
        anchors: torch.Tensor,  # (R, K) unit-norm channel anchors fit once on source codebook
        n_heads: int = 8,
        n_channel_layers: int = 3,
        dropout: float = 0.1,
        route_temp: float = 0.07,
        residual_scale: float = 1.0,
    ) -> None:
        super().__init__()
        n_channels, dim = anchors.shape
        self.n_channels = n_channels
        self.dim = dim
        self.route_temp = float(route_temp)
        self.residual_scale = float(residual_scale)

        self.register_buffer("anchors", F.normalize(anchors.float(), p=2, dim=-1))

        # 1. Contextual chunk encoder + salience scorer (zero-init -> exact zero-shot routing at step 0)
        chunk_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=n_heads,
            dim_feedforward=dim * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.chunk_encoder = nn.TransformerEncoder(chunk_layer, num_layers=2, enable_nested_tensor=False)
        self.chunk_salience = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 1))
        nn.init.zeros_(self.chunk_salience[-1].weight)
        nn.init.zeros_(self.chunk_salience[-1].bias)

        # 2. Multi-head channel-to-chunk cross-attention (extracts multiple distinct clinical facts per chunk)
        self.cross_norm_q = nn.LayerNorm(dim)
        self.cross_norm_kv = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.cross_gate = nn.Parameter(torch.tensor(0.25))

        # 3. Cross-channel reasoning over channel evidence + language-invariant similarity stats
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

        # Zero-init residual heads so epoch 0 starts cleanly on frozen multilingual 2D chunk evidence
        nn.init.zeros_(self.vec_head[-1].weight)
        nn.init.zeros_(self.vec_head[-1].bias)
        nn.init.zeros_(self.gate_delta[-1].weight)
        nn.init.zeros_(self.gate_delta[-1].bias)
        self._last_delta: torch.Tensor | None = None

    def forward(
        self,
        chunks: torch.Tensor,
        mask: torch.Tensor,
        anchors_override: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict 2D Clinical State Matrix U_hat in R^{B x R x K}."""
        valid = mask.bool()
        key_padding_mask = ~valid
        anchors = self.anchors if anchors_override is None else anchors_override

        # 1. Chunk-level contextual clinical salience
        h_chunks = self.chunk_encoder(chunks, src_key_padding_mask=key_padding_mask)
        salience = self.chunk_salience(h_chunks).squeeze(-1)  # (B, M)

        # 2. Frozen-space multilingual channel routing (zero W_V/W_O rotation!)
        sim = torch.einsum("rk,bmk->brm", anchors, chunks)  # (B, R, M)
        route_logits = (sim + salience.unsqueeze(1)) / self.route_temp
        route_logits = route_logits.masked_fill(key_padding_mask.unsqueeze(1), -1e4)
        attn = F.softmax(route_logits, dim=-1)  # (B, R, M)
        evidence = torch.einsum("brm,bmk->brk", attn, chunks)  # (B, R, K)
        evidence = F.normalize(evidence, p=2, dim=-1)

        # Language-invariant per-channel peak & mean chunk cosine similarities
        sim_masked = sim.masked_fill(key_padding_mask.unsqueeze(1), -1.0)
        peak_sim = sim_masked.max(dim=-1, keepdim=True).values  # (B, R, 1)
        denom = valid.float().sum(dim=-1, keepdim=True).clamp(min=1.0).unsqueeze(1)
        mean_sim = (sim * valid.float().unsqueeze(1)).sum(dim=-1, keepdim=True) / denom  # (B, R, 1)

        # 3. Multi-head cross-attention + Cross-channel clinical reasoning
        q = self.cross_norm_q(evidence + anchors.unsqueeze(0))
        kv = self.cross_norm_kv(h_chunks)
        cross_ctx, _ = self.cross_attn(q, kv, kv, key_padding_mask=key_padding_mask, need_weights=False)
        c = self.channel_in_norm(
            evidence + self.cross_gate * cross_ctx + self.sim_proj(torch.cat([peak_sim, mean_sim], dim=-1))
        )
        c = self.channel_encoder(c)

        delta = self.vec_head(self.state_norm(c))
        self._last_delta = delta
        gate = F.softplus(self.peak_scale * (peak_sim - self.peak_shift) + self.gate_delta(c))
        return gate * evidence + self.residual_scale * delta


class _AdaLNBlock(nn.Module):
    """Lightweight DiT-style channel Transformer block conditioned on flow timestep t and note chunks."""

    def __init__(self, dim: int, n_heads: int = 8, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.self_attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.cross_attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.norm3 = nn.LayerNorm(dim, elementwise_affine=False)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
        )
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        nn.init.zeros_(self.ada[-1].weight)
        nn.init.zeros_(self.ada[-1].bias)

    def forward(
        self,
        x: torch.Tensor,
        t_emb: torch.Tensor,
        chunks: torch.Tensor,
        key_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        s1, b1, g1, s3, b3, g3 = self.ada(t_emb).unsqueeze(1).chunk(6, dim=-1)
        h = self.norm1(x) * (1.0 + s1) + b1
        attn_out, _ = self.self_attn(h, h, h, need_weights=False)
        x = x + g1 * attn_out

        h2 = self.norm2(x)
        cross_out, _ = self.cross_attn(h2, chunks, chunks, key_padding_mask=key_padding_mask, need_weights=False)
        x = x + cross_out

        h3 = self.norm3(x) * (1.0 + s3) + b3
        x = x + g3 * self.mlp(h3)
        return x


class StateFieldFlowRefiner(nn.Module):
    """Step 4: Latent 2D Rectified Flow Refiner operating strictly in R^{R x K}.

    Refines the deterministic prior U_init(x) from Step 2 via straight-path velocity matching:
      U_t = (1 - t) * (U_init + sigma * eps) + t * U_0(y)
      v_target = U_0(y) - (U_init + sigma * eps)
    When n_steps == 0 at inference, reduces identically to Step 2 (exact falsification control).
    """

    def __init__(self, dim: int, n_layers: int = 4, n_heads: int = 8, dropout: float = 0.1) -> None:
        super().__init__()
        self.dim = dim
        self.time_mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        self.in_proj = nn.Linear(dim * 2, dim)
        self.blocks = nn.ModuleList([_AdaLNBlock(dim, n_heads=n_heads, dropout=dropout) for _ in range(n_layers)])
        self.out_norm = nn.LayerNorm(dim)
        self.out_proj = nn.Linear(dim, dim)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def _timestep_embedding(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device, dtype=t.dtype) / half)
        args = t.unsqueeze(-1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return self.time_mlp(emb)

    def forward(
        self,
        u_t: torch.Tensor,
        u_init: torch.Tensor,
        t: torch.Tensor,
        chunks: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        t_emb = self._timestep_embedding(t)
        x = self.in_proj(torch.cat([u_t, u_init], dim=-1))
        key_padding_mask = ~mask.bool()
        for blk in self.blocks:
            x = blk(x, t_emb, chunks, key_padding_mask)
        return self.out_proj(self.out_norm(x))

    @torch.inference_mode()
    def sample_euler(
        self,
        u_init: torch.Tensor,
        chunks: torch.Tensor,
        mask: torch.Tensor,
        n_steps: int = 4,
    ) -> torch.Tensor:
        if n_steps <= 0:
            return u_init
        u = u_init.clone()
        dt = 1.0 / n_steps
        bsz = u.shape[0]
        for step in range(n_steps):
            t = torch.full((bsz,), step * dt, device=u.device, dtype=u.dtype)
            vel = self.forward(u, u_init, t, chunks, mask)
            u = u + dt * vel
        return u


class SourceDeconvHead(nn.Module):
    """Frozen precompiled Source-A Self-Normalized Tikhonov operator (single fused tensor-core GEMM)."""

    def __init__(self, chan_indices: tuple[torch.Tensor, ...], chan_weights: tuple[torch.Tensor, ...], n_codes: int) -> None:
        super().__init__()
        self.n_codes = n_codes
        self.n_channels = len(chan_indices)
        dim = 0
        max_m = 1
        for w in chan_weights:
            if w.numel() > 0:
                dim = w.shape[1]
                max_m = max(max_m, w.shape[0])
        self.dim = dim
        self.max_m = max_m

        w_pad = torch.zeros((self.n_channels, max_m, dim), dtype=torch.float32)
        slot_of_code = torch.zeros((n_codes,), dtype=torch.long)
        for r, (idx, w) in enumerate(zip(chan_indices, chan_weights)):
            m = int(idx.numel())
            if m == 0:
                continue
            w_pad[r, :m, :] = w.float()
            slots = r * max_m + torch.arange(m, dtype=torch.long)
            slot_of_code[idx.long()] = slots
        self.register_buffer("w_pad", w_pad)
        self.register_buffer("slot_of_code", slot_of_code)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        # Single fused tensor-core batched matmul: (B, R, K) x (R, M_max, K) -> (B, R, M_max)
        s = states.to(self.w_pad.dtype)
        scores_pad = torch.einsum("brk,rmk->brm", s, self.w_pad)
        return scores_pad.reshape(states.shape[0], -1)[:, self.slot_of_code]


def compute_ocsf_loss(
    u_pred: torch.Tensor,
    u_target: torch.Tensor,
    source_scores: torch.Tensor | None = None,
    y_multi_hot: torch.Tensor | None = None,
    alpha_code: float = 1.0,
    alpha_state: float = 0.25,
    logit_scale: float = 10.0,
    logit_margin: float = 0.50,
    delta: torch.Tensor | None = None,
    beta_anchor: float = 0.01,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Ontology-invariant loss: Calibrated BCE + ZLPR Ranking + Active-Weighted 2D State MSE + Anchor."""
    active_mask = (u_target.norm(dim=-1) > 1e-4).float()  # (B, R)
    weights = 1.0 + 4.0 * active_mask
    per_ch_sq = (u_pred - u_target).pow(2).sum(dim=-1)  # (B, R)
    state_loss = (weights * per_ch_sq).mean()

    anchor_loss = delta.pow(2).sum(dim=-1).mean() if delta is not None else u_pred.new_tensor(0.0)

    if source_scores is None or y_multi_hot is None or alpha_code <= 0.0:
        total = alpha_state * state_loss + beta_anchor * anchor_loss
        return total, {
            "loss_total": float(total.detach()),
            "loss_state": float(state_loss.detach()),
            "loss_code": 0.0,
            "loss_anchor": float(anchor_loss.detach()),
        }

    logits = logit_scale * (source_scores - logit_margin)
    # Calibrated BCE (same objective as PLM-ICD, scaled to O(1) per-note magnitude) + ZLPR ranking
    bce_loss = F.binary_cross_entropy_with_logits(logits, y_multi_hot) * 200.0
    rank_loss = zlpr_loss(logits, y_multi_hot)
    code_loss = bce_loss + 0.15 * rank_loss
    total = alpha_code * code_loss + alpha_state * state_loss + beta_anchor * anchor_loss
    return total, {
        "loss_total": float(total.detach()),
        "loss_state": float(state_loss.detach()),
        "loss_code": float(code_loss.detach()),
        "loss_bce": float(bce_loss.detach()),
        "loss_anchor": float(anchor_loss.detach()),
    }


def build_batch_target_state(
    batch_label_lists: list[list[int]],
    z_a_gpu: torch.Tensor,
    code_ch_gpu: torch.Tensor,
    n_channels: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Vectorized GPU construction of ground-truth 2D state matrix U_0(y) and multi-hot y."""
    bsz = len(batch_label_lists)
    n_codes, dim = z_a_gpu.shape
    dev = z_a_gpu.device
    u_target = torch.zeros((bsz, n_channels, dim), device=dev, dtype=torch.float32)
    y_multi = torch.zeros((bsz, n_codes), device=dev, dtype=torch.float32)

    b_idx: list[int] = []
    c_idx: list[int] = []
    for bi, codes in enumerate(batch_label_lists):
        if not codes:
            continue
        b_idx.extend([bi] * len(codes))
        c_idx.extend(codes)

    if b_idx:
        bt = torch.tensor(b_idx, device=dev, dtype=torch.long)
        ct = torch.tensor(c_idx, device=dev, dtype=torch.long)
        y_multi[bt, ct] = 1.0
        flat_rc = bt * n_channels + code_ch_gpu[ct]
        u_target.view(bsz * n_channels, dim).index_add_(0, flat_rc, z_a_gpu[ct])

    return u_target, y_multi


@torch.inference_mode()
def predict_state_matrices(
    model: StateFieldPredictor,
    chunks_mmap,
    lengths,
    device: torch.device,
    batch_size: int = 128,
    anchors_override=None,
):
    """Predict 2D state matrices U_hat (N, R, K) in float32."""
    import numpy as np

    model.eval()
    n = int(chunks_mmap.shape[0])
    out = np.empty((n, model.n_channels, model.dim), dtype=np.float32)
    anc_t = None
    if anchors_override is not None:
        anc_t = torch.from_numpy(anchors_override.astype(np.float32)).to(device)

    for s in range(0, n, batch_size):
        e = min(s + batch_size, n)
        ch = torch.from_numpy(np.array(chunks_mmap[s:e], dtype=np.float32)).to(device)
        ln = torch.from_numpy(np.asarray(lengths[s:e])).to(device)
        mask = torch.arange(ch.shape[1], device=device)[None, :] < ln[:, None]
        out[s:e] = model(ch, mask, anchors_override=anc_t).cpu().numpy()
    return out

