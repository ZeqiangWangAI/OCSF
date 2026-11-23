"""CPU unit tests for End-to-End Token-Level Bidirectional Diffusion OCSF (src/ocsf/e2e.py)."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from ocsf.e2e import (
    ClosedVocabLAATHead,
    LoRALinear,
    StreamingGPUDeconvScorer,
    TokenChannelStateDecoder,
    TokenLevelOCSFModel,
    inject_upper_layer_lora,
    make_length_bucketed_batches,
)
from ocsf.models import SourceDeconvHead, build_batch_target_state, compute_ocsf_loss
from ocsf.state_matrix import ChannelBank, compile_deconvolution_operator, l2_normalize


class _DummyBlock(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)
        self.gate_proj = nn.Linear(dim, dim, bias=False)
        self.up_proj = nn.Linear(dim, dim, bias=False)
        self.down_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.o_proj(self.q_proj(x) + self.k_proj(x) + self.v_proj(x))
        return x + 0.1 * self.down_proj(torch.nn.functional.silu(self.gate_proj(h)) * self.up_proj(h))


class _DummyOut:
    def __init__(self, h: torch.Tensor) -> None:
        self.last_hidden_state = h


class _DummyBackbone(nn.Module):
    def __init__(self, vocab: int = 100, dim: int = 32, n_layers: int = 4) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab, dim)
        self.layers = nn.ModuleList([_DummyBlock(dim) for _ in range(n_layers)])

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, use_cache: bool = False):
        x = self.embed_tokens(input_ids)
        for layer in self.layers:
            x = layer(x)
        return _DummyOut(x)


def test_upper_layer_lora_freezes_bottom_and_trains_top() -> None:
    bb = _DummyBackbone(dim=32, n_layers=4)
    n_frozen, n_lora = inject_upper_layer_lora(bb, r=4, alpha=8.0, freeze_ratio=0.5)
    assert n_frozen == 2
    assert n_lora == 2 * 7  # 2 top layers x 7 linear modules
    assert not bb.embed_tokens.weight.requires_grad
    assert all(not p.requires_grad for p in bb.layers[0].parameters())
    assert all(not p.requires_grad for p in bb.layers[1].parameters())
    assert isinstance(bb.layers[2].q_proj, LoRALinear)
    assert isinstance(bb.layers[3].down_proj, LoRALinear)
    assert bb.layers[2].q_proj.lora_a.requires_grad
    assert not bb.layers[2].q_proj.base.weight.requires_grad


def test_token_level_ocsf_forward_backward_and_streaming_deconv_exactness() -> None:
    rng = np.random.default_rng(7)
    n_codes, dim, n_ch = 80, 32, 8
    z_a = l2_normalize(rng.normal(size=(n_codes, dim)).astype(np.float32))
    bank = ChannelBank.fit(z_a, n_channels=n_ch, seed=7)
    op = compile_deconvolution_operator(z_a, bank, lam=0.15, self_normalize=True)

    bb = _DummyBackbone(vocab=100, dim=dim, n_layers=4)
    inject_upper_layer_lora(bb, r=4, alpha=8.0, freeze_ratio=0.5)
    dec = TokenChannelStateDecoder(torch.from_numpy(bank.anchors), n_heads=4, n_channel_layers=1)
    model = TokenLevelOCSFModel(bb, dec, apply_tanh=True)
    head = SourceDeconvHead(
        tuple(torch.from_numpy(i) for i in op.chan_indices),
        tuple(torch.from_numpy(w) for w in op.chan_weights),
        n_codes=n_codes,
    )

    ids = torch.randint(1, 99, (3, 20))
    mask = torch.ones((3, 20), dtype=torch.long)
    mask[2, 12:] = 0

    u_hat = model(ids, mask)
    assert u_hat.shape == (3, n_ch, dim)

    # Exact numerical equivalence between SourceDeconvHead (fused GEMM), DeconvolutionOperator, and StreamingGPUDeconvScorer
    s_head = head(u_hat).detach().numpy()
    s_cpu = op.score_batch(u_hat.detach().numpy())
    s_stream = StreamingGPUDeconvScorer(op, torch.device("cpu"), dtype=torch.float32).score_tensor(u_hat)
    assert np.allclose(s_head, s_cpu, atol=1e-4)
    assert np.allclose(s_stream, s_cpu, atol=1e-4)

    u_tgt, y_multi = build_batch_target_state(
        [[0, 5, 11], [2, 19], [7]],
        torch.from_numpy(z_a),
        torch.from_numpy(op.assign),
        n_ch,
    )
    loss, parts = compute_ocsf_loss(u_hat, u_tgt, head(u_hat), y_multi, delta=model.decoder._last_delta)
    assert np.isfinite(parts["loss_total"])
    loss.backward()
    # Confirm gradient reaches both decoder and upper-layer LoRA (and NOT frozen bottom layer)
    assert bb.layers[3].q_proj.lora_b.grad is not None
    assert bb.layers[0].q_proj.weight.grad is None


def test_closed_vocab_laat_and_length_bucketed_batches() -> None:
    head = ClosedVocabLAATHead(dim=32, n_codes=50, proj_dim=16)
    h = torch.randn(2, 15, 32)
    mask = torch.ones((2, 15), dtype=torch.long)
    mask[1, 10:] = 0
    logits = head(h, mask)
    assert logits.shape == (2, 50)
    logits.sum().backward()
    assert head.out_weight.grad is not None

    lens = [100, 500, 120, 480, 900, 110, 510, 880, 300]
    batches = make_length_bucketed_batches(lens, max_tokens=1100, max_batch_size=4, rng=np.random.default_rng(0))
    flat = [i for b in batches for i in b]
    assert sorted(flat) == list(range(len(lens)))
    for b in batches:
        assert len(b) <= 4
        if len(b) > 1:
            assert max(lens[i] for i in b) * len(b) <= 1100
