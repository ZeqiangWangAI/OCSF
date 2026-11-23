"""End-to-End Token-Level Bidirectional Diffusion OCSF Trainer & Zero-Training Transfer Evaluator.

Supports both single-GPU smoke and multi-GPU 4x GH200 DDP via `torchrun --nproc_per_node=4`.
Trains strictly on Source MIMIC-IV ICD-9 train (checkpoint & threshold tau_A selected strictly on
Source ICD-9 val; zero target exposure during training or hyperparameter selection).

Modes:
  - Default (OCSF-E2E): Frozen bottom-50% multilingual layers + LoRA top-50% layers +
    TokenChannelStateDecoder (R=512 channels over all L<=3584 tokens) + frozen Self-Normalized
    Tikhonov Deconvolution readout W_{A,r}. Evaluates B3_2D_dot, Step2_OCSF_deconv, and
    Step3_OCSF_compiled across Source ICD-9 (full + Edin frequent) and all 5 unseen target tables.
  - --closed-vocab-head: Controlled closed-vocabulary PLM-ICD/LAAT reference on the exact same
    backbone & split (evaluated strictly on Source ICD-9 full + Edin frequent; zero-training
    cross-ontology transfer is structurally N/A).
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import datetime
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoModel, AutoTokenizer

from ocsf.compiler import (
    AlignmentStats,
    DRGCompositionalOperator,
    apply_or_groups,
    compile_drg_operator,
    compile_target_operator,
    debias_principal_state,
)
from ocsf.data import (
    DEFAULT_DATA_ROOT,
    iter_source_icd9,
    iter_target_eval,
    load_codebook,
    load_source_subject_ids,
    target_paths,
)
from ocsf.e2e import (
    ClosedVocabLAATHead,
    StreamingGPUDeconvScorer,
    TokenChannelStateDecoder,
    TokenLevelOCSFModel,
    inject_upper_layer_lora,
    make_length_bucketed_batches,
)
from ocsf.encode import BACKBONE_SPECS
from ocsf.metrics import (
    calibrate_source_threshold,
    evaluate_answer_groups,
    evaluate_multilabel,
    paired_bootstrap_ci,
)
from ocsf.models import SourceDeconvHead, build_batch_target_state, compute_ocsf_loss, zlpr_loss
from ocsf.state_matrix import (
    ChannelBank,
    DeconvolutionOperator,
    compile_deconvolution_operator,
    l2_normalize,
)

ALL_TARGETS = ("mimiciv-icd10", "mimiciv-ms-drg", "codiesp", "elcardiocc-2026", "chip-cdn")


def _init_dist() -> tuple[int, int, int]:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", timeout=datetime.timedelta(minutes=60))
        return rank, world, local_rank
    return 0, 1, 0


def _tokenize_texts(tokenizer, texts: list[str], max_length: int) -> list[np.ndarray]:
    """Tokenize documents once into compact int32 token-ID arrays (truncated cleanly at max_length)."""
    out: list[np.ndarray] = []
    # Fast Rust batch tokenization in chunks of 4,096
    for s in range(0, len(texts), 4096):
        enc = tokenizer(
            texts[s : s + 4096],
            add_special_tokens=False,
            truncation=True,
            max_length=max_length,
            padding=False,
            return_attention_mask=False,
        )
        for ids in enc["input_ids"]:
            if not ids:
                ids = [tokenizer.eos_token_id or 0]
            out.append(np.asarray(ids, dtype=np.int32))
    return out


def _collate_ids(
    token_list: list[np.ndarray],
    indices: list[int],
    pad_id: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    seqs = [token_list[i] for i in indices]
    max_len = max(len(s) for s in seqs)
    # Round up to multiple of 8 for Tensor Core efficiency
    max_len = ((max_len + 7) // 8) * 8
    bsz = len(seqs)
    ids = torch.full((bsz, max_len), pad_id, dtype=torch.long)
    mask = torch.zeros((bsz, max_len), dtype=torch.long)
    for bi, seq in enumerate(seqs):
        n = len(seq)
        ids[bi, :n] = torch.from_numpy(seq.astype(np.int64, copy=False))
        mask[bi, :n] = 1
    return ids.to(device, non_blocking=True), mask.to(device, non_blocking=True)


def _build_naive_dot_operator(z_codebook: np.ndarray, bank: ChannelBank) -> DeconvolutionOperator:
    z = l2_normalize(z_codebook.astype(np.float32, copy=False))
    assign = bank.assign_hard(z)
    chan_indices: list[np.ndarray] = []
    chan_weights: list[np.ndarray] = []
    for r in range(bank.n_channels):
        idx = np.flatnonzero(assign == r).astype(np.int32)
        chan_indices.append(idx)
        chan_weights.append(z[idx].astype(np.float32) if idx.size else np.zeros((0, z.shape[1]), dtype=np.float32))
    return DeconvolutionOperator(
        n_codes=z.shape[0],
        n_channels=bank.n_channels,
        dim=z.shape[1],
        assign=assign,
        chan_indices=tuple(chan_indices),
        chan_weights=tuple(chan_weights),
        diag_gains=np.ones(z.shape[0], dtype=np.float32),
    )


def _combine_drg_scores(
    direct: np.ndarray,
    proto_scores: np.ndarray,
    drg_op: DRGCompositionalOperator,
) -> np.ndarray:
    name_to_idx = {name: i for i, name in enumerate(drg_op.prototype_names)}
    proto_mean = proto_scores.mean(axis=1, keepdims=True)
    proto_std = np.maximum(proto_scores.std(axis=1, keepdims=True), 1e-4)
    proto_z = (proto_scores - proto_mean) / proto_std
    direct_std = np.maximum(direct.std(axis=1, keepdims=True), 1e-4)
    out = direct.copy()
    for j, (tier, kind) in enumerate(zip(drg_op.severity_tiers, drg_op.kinds)):
        adj = np.zeros(direct.shape[0], dtype=np.float32)
        if tier in name_to_idx:
            adj += drg_op.severity_weight * proto_z[:, name_to_idx[tier]]
        if kind == "P" and "SURGICAL_P" in name_to_idx:
            adj += drg_op.surgical_weight * proto_z[:, name_to_idx["SURGICAL_P"]]
        elif kind == "M" and "MEDICAL_M" in name_to_idx:
            adj += drg_op.surgical_weight * proto_z[:, name_to_idx["MEDICAL_M"]]
        out[:, j] += adj * direct_std[:, 0]
    return out


class ClosedVocabModel(nn.Module):
    def __init__(self, backbone: nn.Module, head: ClosedVocabLAATHead, apply_tanh: bool) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.apply_tanh = apply_tanh

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        h = out.last_hidden_state
        if self.apply_tanh:
            h = torch.tanh(h)
        h = F.normalize(h.float(), p=2, dim=-1).to(out.last_hidden_state.dtype)
        return self.head(h, attention_mask)


@torch.inference_mode()
def _distributed_eval_source_scores(
    model: nn.Module,
    token_list: list[np.ndarray],
    pad_id: int,
    device: torch.device,
    rank: int,
    world: int,
    max_eval_tokens: int,
    scorer: StreamingGPUDeconvScorer | None = None,
    return_mean_state: bool = False,
    n_channels: int = 0,
    dim: int = 0,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Evaluate all validation notes in parallel across DDP ranks and gather to rank 0."""
    model.eval()
    raw_model = model.module if isinstance(model, DDP) else model
    n_total = len(token_list)
    my_indices = list(range(rank, n_total, world))
    my_lens = [len(token_list[i]) for i in my_indices]
    local_batches = make_length_bucketed_batches(my_lens, max_tokens=max_eval_tokens, max_batch_size=32, rng=None)

    n_codes = scorer.n_codes if scorer is not None else raw_model.head.out_bias.shape[0]
    local_scores = np.empty((len(my_indices), n_codes), dtype=np.float32)
    state_sum = torch.zeros((n_channels, dim), device=device, dtype=torch.float64) if return_mean_state else None
    use_bf16 = device.type == "cuda"

    for lb in local_batches:
        orig_ids = [my_indices[pos] for pos in lb]
        ids, mask = _collate_ids(token_list, orig_ids, pad_id, device)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
            if scorer is not None:
                u = raw_model(ids, mask)
                sc = scorer.score_tensor(u)
                if state_sum is not None:
                    state_sum += u.double().sum(dim=0)
            else:
                sc = raw_model(ids, mask).float().cpu().numpy()
        for k, pos in enumerate(lb):
            local_scores[pos] = sc[k]

    if world == 1:
        mean_state = (state_sum / max(n_total, 1)).cpu().numpy().astype(np.float32) if state_sum is not None else None
        return local_scores, mean_state

    if state_sum is not None:
        dist.all_reduce(state_sum, op=dist.ReduceOp.SUM)
    # Gather local scores by writing temporary shard files or all_gather via GPU/CPU object
    # To avoid NCCL 2GB object limits, write per-rank mmap shard in scratch TMPDIR and merge on rank 0
    tmp_dir = Path(os.environ.get("TMPDIR", "/tmp")) / f"ocsf_eval_{os.getpid() if rank == 0 else 0}"
    dir_list = [str(tmp_dir)]
    dist.broadcast_object_list(dir_list, src=0)
    shared_tmp = Path(dir_list[0])
    if rank == 0:
        shared_tmp.mkdir(parents=True, exist_ok=True)
    dist.barrier()
    np.save(shared_tmp / f"scores_r{rank}.npy", local_scores)
    dist.barrier()

    if rank != 0:
        return None, None

    full_scores = np.empty((n_total, n_codes), dtype=np.float32)
    for r in range(world):
        p = shared_tmp / f"scores_r{r}.npy"
        full_scores[r::world] = np.load(p)
        p.unlink(missing_ok=True)
    try:
        shared_tmp.rmdir()
    except OSError:
        pass
    mean_state = (state_sum / max(n_total, 1)).cpu().numpy().astype(np.float32) if state_sum is not None else None
    return full_scores, mean_state


@torch.inference_mode()
def _stream_eval_target_all_variants(
    model: TokenLevelOCSFModel,
    token_list: list[np.ndarray],
    pad_id: int,
    device: torch.device,
    max_eval_tokens: int,
    dot_op: DeconvolutionOperator,
    deconv_op: DeconvolutionOperator,
    compiled_op: DeconvolutionOperator | DRGCompositionalOperator,
    anchors_override: np.ndarray | None,
    source_mean_state: np.ndarray | None,
    debias_gamma: float,
) -> tuple[dict[str, np.ndarray], float]:
    """Score B3_2D_dot, Step2_OCSF_deconv, and Step3_OCSF_compiled in ONE single backbone pass!"""
    model.eval()
    n = len(token_list)
    n_codes = dot_op.n_codes
    use_bf16 = device.type == "cuda"

    dot_scorer = StreamingGPUDeconvScorer(dot_op, device)
    deconv_scorer = StreamingGPUDeconvScorer(deconv_op, device)
    is_drg = isinstance(compiled_op, DRGCompositionalOperator)
    if is_drg:
        c_direct_scorer = StreamingGPUDeconvScorer(compiled_op.direct_op, device)
        c_proto_scorer = StreamingGPUDeconvScorer(compiled_op.prototype_op, device)
        c_proto_out = np.empty((n, len(compiled_op.prototype_names)), dtype=np.float32)
    else:
        c_direct_scorer = StreamingGPUDeconvScorer(compiled_op, device)
        c_proto_scorer = None
        c_proto_out = None

    anc_ov_t = torch.from_numpy(anchors_override).to(device) if anchors_override is not None else None
    same_anchors = anchors_override is None or np.allclose(
        anchors_override, model.decoder.anchors.detach().cpu().numpy(), atol=1e-5
    )
    mean_st_t = (
        torch.from_numpy(source_mean_state).to(device)
        if (source_mean_state is not None and debias_gamma > 0.0)
        else None
    )

    out_dot = np.empty((n, n_codes), dtype=np.float32)
    out_deconv = np.empty((n, n_codes), dtype=np.float32)
    out_comp = np.empty((n, n_codes), dtype=np.float32)

    batches = make_length_bucketed_batches([len(s) for s in token_list], max_tokens=max_eval_tokens, max_batch_size=32)
    t0 = time.perf_counter()
    for batch in batches:
        ids, mask = _collate_ids(token_list, batch, pad_id, device)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
            h = model.encode_tokens(ids, mask)
            u_base = model.decoder(h, mask)
            u_comp = u_base if same_anchors else model.decoder(h, mask, anchors_override=anc_ov_t)
            if mean_st_t is not None:
                u_comp = u_comp - debias_gamma * mean_st_t.unsqueeze(0)

        sc_dot = dot_scorer.score_tensor(u_base)
        sc_dec = deconv_scorer.score_tensor(u_base)
        sc_comp = c_direct_scorer.score_tensor(u_comp)
        sc_proto = c_proto_scorer.score_tensor(u_comp) if c_proto_scorer is not None else None

        for k, idx in enumerate(batch):
            out_dot[idx] = sc_dot[k]
            out_deconv[idx] = sc_dec[k]
            out_comp[idx] = sc_comp[k]
            if c_proto_out is not None and sc_proto is not None:
                c_proto_out[idx] = sc_proto[k]

    if is_drg and c_proto_out is not None:
        out_comp = _combine_drg_scores(out_comp, c_proto_out, compiled_op)

    infer_ms = (time.perf_counter() - t0) * 1000.0 / max(n, 1)
    return {"B3_2D_dot": out_dot, "Step2_OCSF_deconv": out_deconv, "Step3_OCSF_compiled": out_comp}, infer_ms


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", required=True, choices=sorted(BACKBONE_SPECS.keys()))
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--n-channels", type=int, default=512)
    parser.add_argument("--n-channel-layers", type=int, default=2)
    parser.add_argument("--lam", type=float, default=0.15)
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--lora-alpha", type=float, default=64.0)
    parser.add_argument("--freeze-ratio", type=float, default=0.5)
    parser.add_argument("--max-tokens", type=int, default=3072)
    parser.add_argument("--max-batch-tokens", type=int, default=49152)
    parser.add_argument("--max-eval-tokens", type=int, default=65536)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--lr-lora", type=float, default=2e-4)
    parser.add_argument("--lr-head", type=float, default=1e-3)
    parser.add_argument("--alpha-code", type=float, default=1.0)
    parser.add_argument("--alpha-state", type=float, default=0.35)
    parser.add_argument("--beta-anchor", type=float, default=0.01)
    parser.add_argument("--debias-gamma", type=float, default=0.5)
    parser.add_argument("--closed-vocab-head", action="store_true")
    parser.add_argument("--grad-ckpt", action="store_true")
    parser.add_argument("--resume-from", type=Path, default=None, help="Prior run dir to resume weights & bank from")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--targets", nargs="+", default=list(ALL_TARGETS))
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-val", type=int, default=None)
    parser.add_argument("--limit-target", type=int, default=None)
    args = parser.parse_args()

    rank, world, local_rank = _init_dist()
    dev = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)

    spec = BACKBONE_SPECS[args.backbone]
    cache_root = args.cache_dir / args.backbone
    cb_dir = cache_root / "codebooks"
    run_tag = "closed_laat" if args.closed_vocab_head else f"e2e_R{args.n_channels}_r{args.lora_r}_lam{args.lam}"
    out_root = args.out_dir / args.backbone / run_tag
    if rank == 0:
        out_root.mkdir(parents=True, exist_ok=True)
    if world > 1:
        dist.barrier()

    # 1. Load tokenizer + pre-tokenize Source ICD-9 train & val
    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(spec.hf_id, trust_remote_code=spec.trust_remote_code)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    pad_id = int(tokenizer.pad_token_id)

    src_cb = load_codebook(args.data_root / "benchmarks/mimiciv-2.2/codebooks/icd9-full.json")
    src_map = {e.key: i for i, e in enumerate(src_cb)}
    n_src_codes = len(src_cb)
    icd9_freq_path = args.data_root / "benchmarks/mimiciv-2.2/codebooks/icd9-frequent.json"
    icd9_freq_keys = {e.key for e in load_codebook(icd9_freq_path)} if icd9_freq_path.exists() else None

    train_rows = list(iter_source_icd9("train", args.data_root, limit=args.limit_train))
    val_rows = list(iter_source_icd9("validation", args.data_root, limit=args.limit_val))
    train_labels = [[src_map[k] for k in r.labels if k in src_map] for r in train_rows]
    val_labels = [r.labels for r in val_rows]
    val_gold_idx = [[src_map[k] for k in r.labels if k in src_map] for r in val_rows]

    train_tokens = _tokenize_texts(tokenizer, [r.text for r in train_rows], args.max_tokens)
    val_tokens = _tokenize_texts(tokenizer, [r.text for r in val_rows], args.max_tokens)
    train_lens = [len(s) for s in train_tokens]
    if rank == 0:
        print(
            f"[e2e] Tokenized train={len(train_tokens)} (mean_len={np.mean(train_lens):.0f}), "
            f"val={len(val_tokens)} in {time.perf_counter() - t0:.1f}s | world={world}"
        )

    # 2. Load frozen Source codebook & fit ChannelBank strictly on Source ICD-9 codebook
    z_src = l2_normalize(np.load(cb_dir / "icd9-full.npy").astype(np.float32))
    dim = z_src.shape[1]
    bank_path = out_root / "channel_bank.npz"
    if not args.closed_vocab_head:
        if rank == 0:
            if args.resume_from is not None and (args.resume_from / "channel_bank.npz").exists():
                prev = np.load(args.resume_from / "channel_bank.npz")
                np.savez(bank_path, anchors=prev["anchors"], source_mean=prev["source_mean"])
            else:
                bank = ChannelBank.fit(z_src, n_channels=args.n_channels, seed=args.seed)
                align = AlignmentStats.fit(z_src, use_zca=False)
                np.savez(bank_path, anchors=bank.anchors, source_mean=align.source_mean)
        if world > 1:
            dist.barrier()
        saved = np.load(bank_path)
        bank = ChannelBank(anchors=saved["anchors"])
        align = AlignmentStats(source_mean=saved["source_mean"], zca_matrix=None)
        src_op = compile_deconvolution_operator(z_src, bank, lam=args.lam, self_normalize=True)
        src_head = SourceDeconvHead(
            tuple(torch.from_numpy(idx) for idx in src_op.chan_indices),
            tuple(torch.from_numpy(w) for w in src_op.chan_weights),
            n_codes=n_src_codes,
        ).to(dev)
        src_eval_scorer = StreamingGPUDeconvScorer(src_op, dev)
        z_a_gpu = torch.from_numpy(z_src).to(dev)
        code_ch_gpu = torch.from_numpy(src_op.assign).to(dev)
        if rank == 0:
            print(f"[e2e] ChannelBank R={args.n_channels}, dim={dim}, mean self-gain={src_op.diag_gains.mean():.3f}")
    else:
        bank = align = src_op = src_head = src_eval_scorer = z_a_gpu = code_ch_gpu = None

    # 3. Load HF backbone + inject upper-layer multilingual-preserving LoRA
    backbone = AutoModel.from_pretrained(
        spec.hf_id,
        trust_remote_code=spec.trust_remote_code,
        dtype=torch.bfloat16 if dev.type == "cuda" else torch.float32,
        attn_implementation="sdpa",
    )
    if args.grad_ckpt and hasattr(backbone, "gradient_checkpointing_enable"):
        backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    n_frozen, n_lora = inject_upper_layer_lora(
        backbone, r=args.lora_r, alpha=args.lora_alpha, dropout=0.05, freeze_ratio=args.freeze_ratio
    )
    apply_tanh = spec.family == "pplx_diffusion"

    if args.closed_vocab_head:
        head = ClosedVocabLAATHead(dim=dim, n_codes=n_src_codes)
        model_raw: nn.Module = ClosedVocabModel(backbone, head, apply_tanh=apply_tanh).to(dev)
        head_params = list(model_raw.head.parameters())
    else:
        decoder = TokenChannelStateDecoder(
            torch.from_numpy(bank.anchors),
            n_heads=8,
            n_channel_layers=args.n_channel_layers,
            dropout=0.1,
        )
        model_raw = TokenLevelOCSFModel(backbone, decoder, apply_tanh=apply_tanh).to(dev)
        head_params = list(model_raw.decoder.parameters())

    lora_params = [p for p in model_raw.backbone.parameters() if p.requires_grad]
    if rank == 0:
        n_lora_p = sum(p.numel() for p in lora_params) / 1e6
        n_head_p = sum(p.numel() for p in head_params) / 1e6
        print(
            f"[e2e] Model ready: frozen_bottom_layers={n_frozen}, lora_modules={n_lora} "
            f"(lora={n_lora_p:.2f}M, head={n_head_p:.2f}M trainable params)"
        )

    best_val_r = -1.0
    best_ckpt = out_root / "best_e2e_trainable.pt"
    if args.resume_from is not None:
        prev_ckpt = torch.load(args.resume_from / "best_e2e_trainable.pt", map_location=dev, weights_only=False)
        model_raw.load_state_dict(prev_ckpt["trainable_state"], strict=False)
        best_val_r = float(prev_ckpt.get("val_r_at_y", -1.0))
        if rank == 0:
            torch.save(prev_ckpt, best_ckpt)
            print(f"[e2e] Resumed from {args.resume_from} (prev ep={prev_ckpt.get('epoch')}, best_val_r={best_val_r:.4f})")

    model: nn.Module = (
        DDP(model_raw, device_ids=[local_rank], find_unused_parameters=False) if world > 1 else model_raw
    )

    opt = torch.optim.AdamW(
        [
            {"params": lora_params, "lr": args.lr_lora, "weight_decay": 0.01},
            {"params": head_params, "lr": args.lr_head, "weight_decay": 0.01},
        ]
    )

    # Estimate total optimizer steps for warmup + cosine decay
    est_batches = len(make_length_bucketed_batches(train_lens, args.max_batch_tokens, max_batch_size=48, rng=None))
    steps_per_ep = max(est_batches // (world * args.grad_accum), 1)
    total_steps = max(steps_per_ep * args.epochs, 1)
    warmup_steps = min(max(int(0.05 * total_steps), 10), max(total_steps // 2, 1))

    def _lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = min(float(step - warmup_steps) / float(max(total_steps - warmup_steps, 1)), 1.0)
        return 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * progress))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, _lr_lambda)
    use_bf16 = dev.type == "cuda"

    # 4. Training loop strictly on Source ICD-9 train
    for ep in range(1, args.epochs + 1):
        t_ep = time.perf_counter()
        model.train()
        # Deterministic identical batch list across ranks each epoch, then shard by rank
        ep_rng = np.random.default_rng(args.seed + ep * 997)
        all_batches = make_length_bucketed_batches(
            train_lens, max_tokens=args.max_batch_tokens, max_batch_size=48, rng=ep_rng
        )
        n_usable = (len(all_batches) // world) * world
        rank_batches = all_batches[rank:n_usable:world]

        tot_loss = tot_code = tot_state = 0.0
        n_steps = 0
        opt.zero_grad(set_to_none=True)

        for bi, b_idx in enumerate(rank_batches):
            ids, mask = _collate_ids(train_tokens, b_idx, pad_id, dev)
            is_sync_step = ((bi + 1) % args.grad_accum == 0) or (bi + 1 == len(rank_batches))

            sync_ctx = model.no_sync() if (world > 1 and not is_sync_step) else torch.enable_grad()
            with sync_ctx:
                with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                    if args.closed_vocab_head:
                        y_multi = torch.zeros((len(b_idx), n_src_codes), device=dev, dtype=torch.float32)
                        for row_bi, orig_i in enumerate(b_idx):
                            if train_labels[orig_i]:
                                y_multi[row_bi, train_labels[orig_i]] = 1.0
                        logits = model(ids, mask)
                        loss = F.binary_cross_entropy_with_logits(logits, y_multi)
                        parts = {"loss_total": float(loss.detach()), "loss_code": float(loss.detach()), "loss_state": 0.0}
                    else:
                        u_tgt, y_multi = build_batch_target_state(
                            [train_labels[i] for i in b_idx], z_a_gpu, code_ch_gpu, args.n_channels
                        )
                        u_hat = model(ids, mask)
                        code_scores = src_head(u_hat.float())
                        delta = (model.module if isinstance(model, DDP) else model).decoder._last_delta
                        loss, parts = compute_ocsf_loss(
                            u_hat.float(),
                            u_tgt,
                            code_scores,
                            y_multi,
                            alpha_code=args.alpha_code,
                            alpha_state=args.alpha_state,
                            delta=delta.float() if delta is not None else None,
                            beta_anchor=args.beta_anchor,
                        )
                (loss / args.grad_accum).backward()

            if is_sync_step:
                torch.nn.utils.clip_grad_norm_(lora_params + head_params, 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)

            tot_loss += parts["loss_total"]
            tot_code += parts["loss_code"]
            tot_state += parts["loss_state"]
            n_steps += 1

        # Distributed validation on ALL 21,980 Source ICD-9 val notes
        val_scores, _ = _distributed_eval_source_scores(
            model, val_tokens, pad_id, dev, rank, world, args.max_eval_tokens, scorer=src_eval_scorer
        )
        if rank == 0 and val_scores is not None:
            tau_ep, _ = calibrate_source_threshold(val_scores, val_gold_idx)
            m_full, _ = evaluate_multilabel(val_scores, val_labels, src_map, threshold=tau_ep)
            freq_str = ""
            if icd9_freq_keys is not None:
                m_fq, _ = evaluate_multilabel(
                    val_scores, val_labels, src_map, threshold=tau_ep, subset_keys=icd9_freq_keys
                )
                freq_str = f" | freq6150 F1={m_fq['micro_f1@tau_A']:.4f} R@|y|={m_fq['r@|y|']:.4f} P@8={m_fq['p@8']:.4f}"
            va_r = m_full["r@|y|"]
            marker = ""
            if va_r > best_val_r:
                best_val_r = va_r
                raw_m = model.module if isinstance(model, DDP) else model
                trainable_state = {k: v.cpu() for k, v in raw_m.state_dict().items() if k in {n for n, p in raw_m.named_parameters() if p.requires_grad}}
                torch.save(
                    {"trainable_state": trainable_state, "epoch": ep, "val_r_at_y": va_r, "tau_A": tau_ep},
                    best_ckpt,
                )
                marker = " *best*"
            print(
                f"[e2e] ep={ep:02d}/{args.epochs} loss={tot_loss/max(n_steps,1):.4f} "
                f"(code={tot_code/max(n_steps,1):.4f}, state={tot_state/max(n_steps,1):.4f}) "
                f"full F1={m_full['micro_f1@tau_A']:.4f} R@|y|={va_r:.4f} P@8={m_full['p@8']:.4f} P@15={m_full['p@15']:.4f}"
                f"{freq_str} ({time.perf_counter() - t_ep:.1f}s){marker}"
            )
        if world > 1:
            dist.barrier()

    # 5. Reload best Source-val checkpoint and run final evaluation
    ckpt = torch.load(best_ckpt, map_location=dev, weights_only=False)
    raw_m = model.module if isinstance(model, DDP) else model
    raw_m.load_state_dict(ckpt["trainable_state"], strict=False)
    raw_m.eval()

    if args.closed_vocab_head:
        val_scores, _ = _distributed_eval_source_scores(
            model, val_tokens, pad_id, dev, rank, world, args.max_eval_tokens, scorer=None
        )
        if rank == 0 and val_scores is not None:
            tau_a, _ = calibrate_source_threshold(val_scores, val_gold_idx)
            m_full, pn_full = evaluate_multilabel(val_scores, val_labels, src_map, threshold=tau_a)
            m_full["tau_A"] = tau_a
            res = {"source_icd9_val": m_full}
            if icd9_freq_keys is not None:
                m_fq, pn_fq = evaluate_multilabel(
                    val_scores, val_labels, src_map, threshold=tau_a, subset_keys=icd9_freq_keys
                )
                res["source_icd9_val_frequent"] = m_fq
            (out_root / "closed_vocab_summary.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
            print(f"[closed-vocab] Saved -> {out_root / 'closed_vocab_summary.json'}")
            for k, m in res.items():
                print(
                    f"  {k:<26} F1={m['micro_f1@tau_A']:.4f} R@|y|={m['r@|y|']:.4f} "
                    f"P@8={m['p@8']:.4f} P@15={m['p@15']:.4f} MRR={m['mrr']:.4f}"
                )
        if world > 1:
            dist.destroy_process_group()
        return

    # OCSF-E2E: distributed Source val scores + source_mean_state (for principal-diagnosis debiasing)
    src_dot_op = _build_naive_dot_operator(z_src, bank)
    src_dot_scorer = StreamingGPUDeconvScorer(src_dot_op, dev)
    src_scores_deconv, source_mean_state = _distributed_eval_source_scores(
        model,
        val_tokens,
        pad_id,
        dev,
        rank,
        world,
        args.max_eval_tokens,
        scorer=src_eval_scorer,
        return_mean_state=True,
        n_channels=args.n_channels,
        dim=dim,
    )
    src_scores_dot, _ = _distributed_eval_source_scores(
        model, val_tokens, pad_id, dev, rank, world, args.max_eval_tokens, scorer=src_dot_scorer
    )

    if rank == 0:
        assert src_scores_deconv is not None and src_scores_dot is not None and source_mean_state is not None
        summary: dict[str, dict[str, dict[str, float]]] = {
            "B3_2D_dot": {},
            "Step2_OCSF_deconv": {},
            "Step3_OCSF_compiled": {},
        }
        tau_by_var: dict[str, float] = {}
        for v_name, sc in [
            ("B3_2D_dot", src_scores_dot),
            ("Step2_OCSF_deconv", src_scores_deconv),
            ("Step3_OCSF_compiled", src_scores_deconv),
        ]:
            pn_dir = out_root / "per_note" / v_name
            pn_dir.mkdir(parents=True, exist_ok=True)
            tau_a, _ = calibrate_source_threshold(sc, val_gold_idx)
            tau_by_var[v_name] = tau_a
            m_src, pn_src = evaluate_multilabel(sc, val_labels, src_map, threshold=tau_a)
            m_src["tau_A"] = tau_a
            summary[v_name]["source_icd9_val"] = m_src
            np.save(pn_dir / "source_icd9_val.npy", pn_src)
            if icd9_freq_keys is not None:
                m_fq, pn_fq = evaluate_multilabel(
                    sc, val_labels, src_map, threshold=tau_a, subset_keys=icd9_freq_keys
                )
                summary[v_name]["source_icd9_val_frequent"] = m_fq
                np.save(pn_dir / "source_icd9_val_frequent.npy", pn_fq)

        # Evaluate all 5 frozen non-test targets on rank 0 in ONE streaming backbone pass per target
        source_subjects = load_source_subject_ids(args.data_root)
        drg_protos = dict(np.load(cb_dir / "drg_prototypes.npz")) if (cb_dir / "drg_prototypes.npz").exists() else {}

        for tgt in args.targets:
            tpaths = target_paths(tgt, args.data_root)
            cb = load_codebook(tpaths["codebook"])
            code_map = {e.key: i for i, e in enumerate(cb)}
            rows = list(iter_target_eval(tgt, args.data_root, source_subjects=source_subjects, limit=args.limit_target))
            z_t = np.load(cb_dir / f"{tgt}.npy")
            tgt_tokens = _tokenize_texts(tokenizer, [r.text for r in rows], args.max_tokens)

            t_c = time.perf_counter()
            dot_op = _build_naive_dot_operator(z_t, bank)
            deconv_op = compile_deconvolution_operator(z_t, bank, lam=args.lam, self_normalize=True)
            if tgt == "mimiciv-ms-drg" and drg_protos:
                comp_op: DeconvolutionOperator | DRGCompositionalOperator = compile_drg_operator(
                    cb, z_t, drg_protos, bank, cms_dir=tpaths["cms_dir"], alignment=None, lam=args.lam
                )
            else:
                comp_op = deconv_op
            compile_sec = time.perf_counter() - t_c

            is_principal = tgt in {"mimiciv-ms-drg", "elcardiocc-2026"}
            var_scores, infer_ms = _stream_eval_target_all_variants(
                raw_m,
                tgt_tokens,
                pad_id,
                dev,
                args.max_eval_tokens,
                dot_op=dot_op,
                deconv_op=deconv_op,
                compiled_op=comp_op,
                anchors_override=None,
                source_mean_state=source_mean_state if is_principal else None,
                debias_gamma=args.debias_gamma if is_principal else 0.0,
            )

            icd10_freq_keys = (
                {e.key for e in load_codebook(tpaths["frequent_codebook"])}
                if (tgt == "mimiciv-icd10" and tpaths["frequent_codebook"].exists())
                else None
            )
            for v_name, sc in var_scores.items():
                pn_dir = out_root / "per_note" / v_name
                cmap_v = code_map
                if tgt == "codiesp" and v_name == "Step3_OCSF_compiled" and tpaths["groups"].exists():
                    sc, cmap_v = apply_or_groups(sc, code_map, tpaths["groups"])

                if is_principal:
                    m, pn = evaluate_answer_groups(sc, [r.label_groups for r in rows], cmap_v)
                else:
                    m, pn = evaluate_multilabel(sc, [r.labels for r in rows], cmap_v, threshold=tau_by_var[v_name])
                m["compile_sec"] = compile_sec
                m["infer_ms_per_note"] = infer_ms
                summary[v_name][tgt] = m
                np.save(pn_dir / f"{tgt}.npy", pn)

                if icd10_freq_keys is not None:
                    rare_keys = set(cmap_v.keys()) - icd10_freq_keys
                    mf, pnf = evaluate_multilabel(
                        sc, [r.labels for r in rows], cmap_v, threshold=tau_by_var[v_name], subset_keys=icd10_freq_keys
                    )
                    mr, pnr = evaluate_multilabel(
                        sc, [r.labels for r in rows], cmap_v, threshold=tau_by_var[v_name], subset_keys=rare_keys
                    )
                    summary[v_name]["mimiciv-icd10-frequent"] = mf
                    summary[v_name]["mimiciv-icd10-rare"] = mr
                    np.save(pn_dir / "mimiciv-icd10-frequent.npy", pnf)
                    np.save(pn_dir / "mimiciv-icd10-rare.npy", pnr)
            print(f"[e2e-eval] Finished target={tgt} (N={len(rows)}, compile={compile_sec:.2f}s, infer={infer_ms:.2f}ms/note)")

        # Paired bootstrap CIs
        bootstrap_report: dict[str, dict[str, dict[str, float]]] = {}
        comparisons = [
            ("Step2_minus_B3_dot", out_root / "per_note/B3_2D_dot", out_root / "per_note/Step2_OCSF_deconv"),
            ("Step3_minus_Step2", out_root / "per_note/Step2_OCSF_deconv", out_root / "per_note/Step3_OCSF_compiled"),
        ]
        for comp_name, base_dir, cand_dir in comparisons:
            bootstrap_report[comp_name] = {}
            for p_cand in sorted(cand_dir.glob("*.npy")):
                p_base = base_dir / p_cand.name
                if p_base.exists():
                    bootstrap_report[comp_name][p_cand.stem] = asdict(
                        paired_bootstrap_ci(np.load(p_base), np.load(p_cand))
                    )

        (out_root / "e2e_ocsf_summary.json").write_text(
            json.dumps({"results": summary, "bootstrap": bootstrap_report}, indent=2), encoding="utf-8"
        )
        print(f"[e2e] Saved summary -> {out_root / 'e2e_ocsf_summary.json'}")
        for v_name, res in summary.items():
            print(f"\n=== E2E {args.backbone} | {v_name} ===")
            for bench, m in res.items():
                if "acc@1" in m:
                    s = f"acc@1={m['acc@1']:.4f} acc@5={m['acc@5']:.4f}"
                else:
                    s = f"R@|y|={m['r@|y|']:.4f} F1@tauA={m.get('micro_f1@tau_A', 0.0):.4f} P@8={m['p@8']:.4f} P@15={m['p@15']:.4f}"
                print(f"  {bench:<26} {s}  MRR={m['mrr']:.4f}")

    if world > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
