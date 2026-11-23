"""Comprehensive Reviewer-Response Evaluation Suite (Strictly Non-Test Splits).

Executes five targeted empirical experiments directly addressing peer-review questions:
  R1. End-to-End Inference Latency & Peak VRAM Profile (0.6B & 4B across L in {512, 1024, 2048, 3072}, bs in {1, 8, 16}).
  R2. Label-Description Perturbation & Embedding-Noise Robustness (Clean, 50% Truncation, 25% Word Drop,
      Gaussian noise sigma in {0.10, 0.20}) comparing B3_2D_dot, Unnorm Deconv, and SelfNorm Deconv on
      MIMIC ICD-10 (EN), CodiEsp (ES), and CHIP-CDN (ZH).
  R3. Calibration-Free Metrics (Mean-Note AP / Micro-AUPRC) & Target-Agnostic Zero-Label Thresholds
      (source tau_A vs theoretical unit-response midpoint 0.50 vs source-matched cardinality).
  R4. Channel Partitioning Strategy Ablation (Spherical k-Means vs Hierarchical Ward vs Random Anchors
      on real pplx-0.6b codebook geometry + zero-retrain Soft Top-2 target assignment).
  R5. Attention Directionality & Length-Stratified Diagnostics (exact same-weight is_causal=False vs True
      intervention on frozen pplx-0.6b; short/medium/long note-length terciles for pplx vs qwen; per-CMS-version
      MS-DRG v25..v37 consistency breakdown).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import average_precision_score
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from ocsf.compiler import (
    DRGCompositionalOperator,
    apply_or_groups,
    compile_drg_operator,
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
    StreamingGPUDeconvScorer,
    TokenChannelStateDecoder,
    TokenLevelOCSFModel,
    inject_upper_layer_lora,
    make_length_bucketed_batches,
)
from ocsf.encode import BACKBONE_SPECS, ClinicalEncoder
from ocsf.metrics import (
    compute_threshold_micro_f1,
    evaluate_answer_groups,
    evaluate_multilabel,
)
from ocsf.state_matrix import (
    ChannelBank,
    compile_deconvolution_operator,
    encode_ground_truth_state,
    l2_normalize,
)
from run_train_e2e_ocsf import (
    _build_naive_dot_operator,
    _collate_ids,
    _combine_drg_scores,
    _tokenize_texts,
)


def _load_drg_prototypes(cb_dir: Path) -> dict[str, np.ndarray]:
    p = cb_dir / "drg_prototypes.npz"
    if not p.exists():
        return {}
    data = np.load(p)
    return {k: data[k].astype(np.float32) for k in data.files}


def _load_ocsf_model(
    backbone_name: str,
    run_dir: Path,
    dev: torch.device,
    lora_r: int = 32,
    lora_alpha: float = 64.0,
    freeze_ratio: float = 0.5,
) -> tuple[TokenLevelOCSFModel, ChannelBank, object, float]:
    spec = BACKBONE_SPECS[backbone_name]
    padding_side = "left" if spec.family == "qwen_causal" else "right"
    tok = AutoTokenizer.from_pretrained(spec.hf_id, trust_remote_code=spec.trust_remote_code, padding_side=padding_side)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id

    saved_bank = np.load(run_dir / "channel_bank.npz")
    bank = ChannelBank(anchors=saved_bank["anchors"].astype(np.float32))
    ckpt = torch.load(run_dir / "best_e2e_trainable.pt", map_location="cpu", weights_only=False)
    tau_a = 0.50
    sum_path = run_dir / "e2e_ocsf_summary.json"
    if sum_path.exists():
        sdoc = json.loads(sum_path.read_text(encoding="utf-8"))
        tau_a = float(sdoc["results"]["Step2_OCSF_deconv"]["source_icd9_val"].get("tau_A", 0.50))
    use_bf16 = dev.type == "cuda"
    backbone = AutoModel.from_pretrained(
        spec.hf_id,
        trust_remote_code=spec.trust_remote_code,
        dtype=torch.bfloat16 if use_bf16 else torch.float32,
        attn_implementation="sdpa",
    )
    inject_upper_layer_lora(backbone, r=lora_r, alpha=lora_alpha, dropout=0.0, freeze_ratio=freeze_ratio)
    decoder = TokenChannelStateDecoder(torch.from_numpy(bank.anchors), n_heads=8, n_channel_layers=2, dropout=0.0)
    model = TokenLevelOCSFModel(backbone, decoder, apply_tanh=(spec.family == "pplx_diffusion")).to(dev)
    res = model.load_state_dict(ckpt["trainable_state"], strict=False)
    if res.unexpected_keys:
        raise RuntimeError(f"Unexpected checkpoint keys: {res.unexpected_keys[:5]}")
    model.eval()
    return model, bank, tok, tau_a


@torch.inference_mode()
def run_r1_latency_vram_profile(
    run_dir_06: Path,
    run_dir_4b: Path | None,
    cache_dir: Path,
    dev: torch.device,
) -> dict:
    """R1: Exact end-to-end wall-clock latency & peak VRAM decomposition for 0.6B and 4B."""
    results: dict = {}
    targets_to_profile = [("pplx-0.6b", run_dir_06)]
    if run_dir_4b is not None and (run_dir_4b / "best_e2e_trainable.pt").exists():
        targets_to_profile.append(("pplx-4b", run_dir_4b))

    for bname, rdir in targets_to_profile:
        print(f"[R1] Profiling {bname} from {rdir} ...", flush=True)
        torch.cuda.empty_cache()
        model, bank, tok, _ = _load_ocsf_model(bname, rdir, dev)
        cb_dir = cache_dir / bname / "codebooks"
        op_icd10 = compile_deconvolution_operator(np.load(cb_dir / "mimiciv-icd10.npy"), bank, lam=0.15)
        op_codi = compile_deconvolution_operator(np.load(cb_dir / "codiesp.npy"), bank, lam=0.15)
        sc_icd10 = StreamingGPUDeconvScorer(op_icd10, dev)
        sc_codi = StreamingGPUDeconvScorer(op_codi, dev)

        grid = []
        vocab_hi = min(int(getattr(tok, "vocab_size", 50000)), 50000)
        for seq_len in (512, 1024, 2048, 3072):
            for bs in (1, 8, 16):
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(dev)
                ids = torch.randint(100, vocab_hi, (bs, seq_len), device=dev, dtype=torch.long)
                mask = torch.ones((bs, seq_len), device=dev, dtype=torch.long)

                for _ in range(2):
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        h = model.encode_tokens(ids, mask)
                        u = model.decoder(h, mask)
                    _ = sc_icd10.score_tensor(u)
                torch.cuda.synchronize()

                n_iters = 5
                t_bb = t_dec = t_r10 = t_rcodi = 0.0
                for _ in range(n_iters):
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        h = model.encode_tokens(ids, mask)
                    torch.cuda.synchronize()
                    t1 = time.perf_counter()
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        u = model.decoder(h, mask)
                    torch.cuda.synchronize()
                    t2 = time.perf_counter()
                    _ = sc_icd10.score_tensor(u)
                    torch.cuda.synchronize()
                    t3 = time.perf_counter()
                    _ = sc_codi.score_tensor(u)
                    torch.cuda.synchronize()
                    t4 = time.perf_counter()
                    t_bb += t1 - t0
                    t_dec += t2 - t1
                    t_r10 += t3 - t2
                    t_rcodi += t4 - t3

                peak_gb = float(torch.cuda.max_memory_allocated(dev)) / (1024.0 ** 3)
                denom = n_iters * bs
                bb_ms = 1000.0 * t_bb / denom
                dec_ms = 1000.0 * t_dec / denom
                r10_ms = 1000.0 * t_r10 / denom
                rco_ms = 1000.0 * t_rcodi / denom
                tot10_ms = bb_ms + dec_ms + r10_ms
                totco_ms = bb_ms + dec_ms + rco_ms
                row = {
                    "seq_len": seq_len,
                    "batch_size": bs,
                    "backbone_ms_per_note": round(bb_ms, 3),
                    "decoder_ms_per_note": round(dec_ms, 3),
                    "readout_icd10_ms_per_note": round(r10_ms, 4),
                    "readout_codiesp_ms_per_note": round(rco_ms, 4),
                    "total_icd10_ms_per_note": round(tot10_ms, 3),
                    "total_codiesp_ms_per_note": round(totco_ms, 3),
                    "throughput_icd10_notes_per_sec": round(1000.0 / max(tot10_ms, 1e-6), 2),
                    "peak_vram_gb": round(peak_gb, 2),
                }
                grid.append(row)
                print(
                    f"  [{bname}] L={seq_len:4d} bs={bs:2d} | bb={bb_ms:6.2f}ms dec={dec_ms:5.2f}ms "
                    f"r10={r10_ms:.3f}ms tot10={tot10_ms:6.2f}ms ({row['throughput_icd10_notes_per_sec']:6.1f} notes/s) "
                    f"VRAM={peak_gb:5.2f}GB",
                    flush=True,
                )
        results[bname] = grid
        del model, sc_icd10, sc_codi
        torch.cuda.empty_cache()
    return results


def _perturb_text(text: str, mode: str, rng: np.random.Generator) -> str:
    s = text.strip()
    if mode == "trunc_50pct":
        keep = max(6, len(s) // 2)
        return s[:keep].strip()
    if mode == "word_drop_25pct":
        words = s.split()
        if len(words) >= 4:
            kept = [w for w in words if rng.random() > 0.25]
            return " ".join(kept) if kept else words[0]
        chars = list(s)
        if len(chars) >= 4:
            kept = [c for c in chars if rng.random() > 0.25]
            return "".join(kept) if kept else chars[0]
        return s
    raise ValueError(mode)


@torch.inference_mode()
def _score_multi_ops(
    model: TokenLevelOCSFModel,
    token_seqs: list[np.ndarray],
    pad_id: int,
    dev: torch.device,
    max_eval_tokens: int,
    scorers: dict[str, StreamingGPUDeconvScorer],
) -> dict[str, np.ndarray]:
    n = len(token_seqs)
    use_bf16 = dev.type == "cuda"
    outs = {k: np.empty((n, sc.n_codes), dtype=np.float32) for k, sc in scorers.items()}
    batches = make_length_bucketed_batches([len(s) for s in token_seqs], max_tokens=max_eval_tokens, max_batch_size=32)
    for batch in batches:
        ids_t, mask_t = _collate_ids(token_seqs, batch, pad_id, dev)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
            u = model(ids_t, mask_t)
        for name, sc in scorers.items():
            vals = sc.score_tensor(u)
            for k, idx in enumerate(batch):
                outs[name][idx] = vals[k]
    return outs


def _compute_calibration_free_ap(
    scores: np.ndarray,
    gold_in_vocab: list[list[int]],
    gold_total_counts: list[int],
    top_k_micro: int = 200,
) -> dict[str, float]:
    """Compute exact Mean-Note Average Precision (mAP, OOV-preserving) & Top-200 Micro-AUPRC lower bound."""
    n, n_codes = scores.shape
    note_aps: list[float] = []
    micro_y_parts: list[np.ndarray] = []
    micro_s_parts: list[np.ndarray] = []
    total_pos = 0
    captured_pos = 0

    use_exact_micro = (n * n_codes) <= 40_000_000
    for i in range(n):
        n_gold = gold_total_counts[i]
        if n_gold <= 0:
            continue
        g_valid = gold_in_vocab[i]
        total_pos += n_gold
        if not g_valid:
            note_aps.append(0.0)
            continue
        row = scores[i]
        g_set = set(g_valid)
        pos_scores = np.sort(row[g_valid])[::-1]
        ranks = np.array([int(np.sum(row > s)) + 1 for s in pos_scores], dtype=np.float64)
        for j in range(1, len(ranks)):
            if ranks[j] <= ranks[j - 1]:
                ranks[j] = ranks[j - 1] + 1.0
        prec_at_pos = np.arange(1, len(g_valid) + 1, dtype=np.float64) / ranks
        note_aps.append(float(np.sum(prec_at_pos) / float(n_gold)))

        if use_exact_micro:
            y_row = np.zeros(n_codes, dtype=np.int8)
            y_row[g_valid] = 1
            micro_y_parts.append(y_row)
            micro_s_parts.append(row)
            captured_pos += len(g_valid)
        else:
            k = min(top_k_micro, n_codes)
            top_idx = np.argpartition(-row, k - 1)[:k]
            y_k = np.array([1 if c in g_set else 0 for c in top_idx], dtype=np.int8)
            micro_y_parts.append(y_k)
            micro_s_parts.append(row[top_idx])
            captured_pos += int(y_k.sum())

    mean_note_ap = float(np.mean(note_aps)) if note_aps else 0.0
    y_all = np.concatenate(micro_y_parts) if micro_y_parts else np.zeros(1, dtype=np.int8)
    s_all = np.concatenate(micro_s_parts) if micro_s_parts else np.zeros(1, dtype=np.float32)
    raw_micro_ap = float(average_precision_score(y_all, s_all)) if int(y_all.sum()) > 0 else 0.0
    recall_cap = float(captured_pos) / max(float(total_pos), 1.0)
    return {
        "mean_note_ap": round(mean_note_ap, 4),
        "micro_auprc_lb": round(raw_micro_ap * recall_cap, 4),
        "top200_pos_capture": round(recall_cap, 4),
    }


def _zero_label_cardinality_f1(
    scores: np.ndarray,
    gold_in_vocab: list[list[int]],
    total_gold_count: int,
    k_fixed: int,
) -> float:
    tp = fp = 0
    k = min(max(k_fixed, 1), scores.shape[1])
    for i, g in enumerate(gold_in_vocab):
        top = set(np.argpartition(-scores[i], k - 1)[:k].tolist())
        hit = len(top.intersection(g))
        tp += hit
        fp += k - hit
    denom = (tp + fp) + total_gold_count
    return float(2 * tp / max(denom, 1))


def run_r2_robustness(
    model: TokenLevelOCSFModel,
    bank: ChannelBank,
    tok: object,
    tau_a: float,
    cache_dir: Path,
    data_root: Path,
    src_subjs: set[int],
    dev: torch.device,
) -> dict:
    """R2: Description truncation / word-drop / embedding-noise robustness stress-test."""
    print("[R2] Running label-description & embedding-noise robustness stress-test...", flush=True)
    cb_enc = ClinicalEncoder("pplx-0.6b", device=str(dev))
    pad_id = int(tok.pad_token_id)
    rng = np.random.default_rng(42)
    out: dict = {}

    for tgt in ("mimiciv-icd10", "codiesp", "chip-cdn"):
        tpaths = target_paths(tgt, data_root)
        cb = load_codebook(tpaths["codebook"])
        exs = list(iter_target_eval(tgt, data_root, source_subjects=src_subjs, limit=2000))
        k2i = {e.key: i for i, e in enumerate(cb)}
        gold_keys = [ex.labels for ex in exs]
        tok_seqs = _tokenize_texts(tok, [e.text for e in exs], max_length=3072)

        z_clean = l2_normalize(np.load(cache_dir / "pplx-0.6b" / "codebooks" / f"{tgt}.npy").astype(np.float32))
        perturb_z: dict[str, np.ndarray] = {"clean": z_clean}
        for pmode in ("trunc_50pct", "word_drop_25pct"):
            p_texts = [_perturb_text(e.prompt_text, pmode, rng) for e in cb]
            perturb_z[pmode] = l2_normalize(cb_enc.encode_codebook_texts(p_texts, batch_size=256))
        for sigma in (0.10, 0.20):
            noise = rng.normal(0.0, sigma, size=z_clean.shape).astype(np.float32)
            perturb_z[f"emb_noise_{sigma:.2f}"] = l2_normalize(z_clean + noise)

        tgt_res: dict = {}
        for pname, zp in perturb_z.items():
            ops = {
                "B3_2D_dot": _build_naive_dot_operator(zp, bank),
                "Unnorm_deconv": compile_deconvolution_operator(zp, bank, lam=0.15, self_normalize=False),
                "SelfNorm_deconv": compile_deconvolution_operator(zp, bank, lam=0.15, self_normalize=True),
            }
            scorers = {k: StreamingGPUDeconvScorer(op, dev) for k, op in ops.items()}
            sc_map = _score_multi_ops(model, tok_seqs, pad_id, dev, 65536, scorers)
            tgt_res[pname] = {}
            for vname, sc in sc_map.items():
                m, _ = evaluate_multilabel(sc, gold_keys, k2i, threshold=tau_a)
                tgt_res[pname][vname] = {
                    "r@|y|": round(m["r@|y|"], 4),
                    "micro_f1@tau_A": round(m["micro_f1@tau_A"], 4),
                    "mrr": round(m["mrr"], 4),
                }
            print(
                f"  [R2:{tgt} | {pname:<16}] Dot={tgt_res[pname]['B3_2D_dot']['r@|y|']:.4f} "
                f"Unnorm={tgt_res[pname]['Unnorm_deconv']['r@|y|']:.4f} "
                f"SelfNorm={tgt_res[pname]['SelfNorm_deconv']['r@|y|']:.4f}",
                flush=True,
            )
        out[tgt] = tgt_res

    del cb_enc
    torch.cuda.empty_cache()
    return out


def run_r3_calibration_and_r5_drg_versions(
    model: TokenLevelOCSFModel,
    bank: ChannelBank,
    tok: object,
    tau_a: float,
    cache_dir: Path,
    data_root: Path,
    src_subjs: set[int],
    dev: torch.device,
) -> tuple[dict, dict]:
    """R3: Calibration-free mAP/AUPRC & zero-label plug-in thresholds + R5c per-CMS-version MS-DRG consistency."""
    print("[R3/R5c] Running calibration-free AUPRC, zero-label thresholds & MS-DRG version audit...", flush=True)
    pad_id = int(tok.pad_token_id)
    cb_dir = cache_dir / "pplx-0.6b" / "codebooks"

    src_cb = load_codebook(data_root / "benchmarks/mimiciv-2.2/codebooks/icd9-full.json")
    src_val = list(iter_source_icd9("validation", data_root))
    mean_src_k = int(round(float(np.mean([len(ex.labels) for ex in src_val]))))
    z_src = np.load(cb_dir / "icd9-full.npy").astype(np.float32)

    val_toks = _tokenize_texts(tok, [e.text for e in src_val[:4000]], max_length=3072)
    batches = make_length_bucketed_batches([len(s) for s in val_toks], max_tokens=65536, max_batch_size=32)
    state_sum = torch.zeros((bank.n_channels, bank.dim), device=dev, dtype=torch.float32)
    with torch.inference_mode():
        for b in batches:
            ids_t, mask_t = _collate_ids(val_toks, b, pad_id, dev)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                u = model(ids_t, mask_t)
            state_sum += u.float().sum(dim=0)
    source_mean_state = (state_sum / float(len(val_toks))).cpu().numpy().astype(np.float32)

    cal_res: dict = {"mean_source_cardinality": mean_src_k, "tau_A": round(tau_a, 4), "benchmarks": {}}
    drg_versions_res: dict = {}
    drg_protos = _load_drg_prototypes(cb_dir)

    all_evals = [("source_icd9_val_4k", src_cb, z_src, src_val[:4000])]
    for tgt in ("mimiciv-icd10", "mimiciv-ms-drg", "codiesp", "elcardiocc-2026", "chip-cdn"):
        tp = target_paths(tgt, data_root)
        cb = load_codebook(tp["codebook"])
        exs = list(iter_target_eval(tgt, data_root, source_subjects=src_subjs))
        all_evals.append((tgt, cb, np.load(cb_dir / f"{tgt}.npy").astype(np.float32), exs))

    for name, cb, z_t, exs in all_evals:
        k2i = {e.key: i for i, e in enumerate(cb)}
        toks = _tokenize_texts(tok, [e.text for e in exs], max_length=3072)
        dot_op = _build_naive_dot_operator(z_t, bank)
        dec_op = compile_deconvolution_operator(z_t, bank, lam=0.15, self_normalize=True)
        scorers = {
            "B3_2D_dot": StreamingGPUDeconvScorer(dot_op, dev),
            "Step2_deconv": StreamingGPUDeconvScorer(dec_op, dev),
        }
        sc_map = _score_multi_ops(model, toks, pad_id, dev, 65536, scorers)

        if name == "mimiciv-ms-drg" and drg_protos:
            tp = target_paths(name, data_root)
            drg_op = compile_drg_operator(cb, z_t, drg_protos, bank, cms_dir=tp["cms_dir"], alignment=None, lam=0.15)
            d_sc = StreamingGPUDeconvScorer(drg_op.direct_op, dev)
            p_sc = StreamingGPUDeconvScorer(drg_op.prototype_op, dev)
            ms_t = torch.from_numpy(source_mean_state).to(dev)
            dir_all = np.empty((len(exs), d_sc.n_codes), dtype=np.float32)
            pro_all = np.empty((len(exs), p_sc.n_codes), dtype=np.float32)
            bs = make_length_bucketed_batches([len(s) for s in toks], max_tokens=65536, max_batch_size=32)
            with torch.inference_mode():
                for b in bs:
                    ids_t, mask_t = _collate_ids(toks, b, pad_id, dev)
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        u = model(ids_t, mask_t) - 0.5 * ms_t.unsqueeze(0)
                    dv = d_sc.score_tensor(u)
                    pv = p_sc.score_tensor(u)
                    for k, idx in enumerate(b):
                        dir_all[idx] = dv[k]
                        pro_all[idx] = pv[k]
            step3_drg = _combine_drg_scores(dir_all, pro_all, drg_op)
            sc_map["Step3_compiled"] = step3_drg

            ver_buckets: dict[str, list[int]] = {}
            for idx, ex in enumerate(exs):
                vers = set()
                for grp in ex.label_groups:
                    for k in grp:
                        parts = k.split(":")
                        if len(parts) >= 3 and parts[-1].startswith("v"):
                            vers.add(parts[-1])
                for v in sorted(vers):
                    ver_buckets.setdefault(v, []).append(idx)
            for v in sorted(ver_buckets.keys()):
                idxs = ver_buckets[v]
                if len(idxs) < 50:
                    continue
                grps = [exs[i].label_groups for i in idxs]
                m2, _ = evaluate_answer_groups(sc_map["Step2_deconv"][idxs], grps, k2i)
                m3, _ = evaluate_answer_groups(step3_drg[idxs], grps, k2i)
                drg_versions_res[v] = {
                    "n_notes": len(idxs),
                    "step2_acc1": round(m2["acc@1"], 4),
                    "step3_acc1": round(m3["acc@1"], 4),
                    "step2_acc5": round(m2["acc@5"], 4),
                    "step3_acc5": round(m3["acc@5"], 4),
                    "delta_acc5": round(m3["acc@5"] - m2["acc@5"], 4),
                }
            print(f"  [R5c] MS-DRG versions audited: {len(drg_versions_res)} versions", flush=True)

        if name in {"mimiciv-ms-drg", "elcardiocc-2026"}:
            flat_in = [sorted({k2i[k] for grp in ex.label_groups for k in grp if k in k2i}) for ex in exs]
            flat_cnt = [len(g) for g in flat_in]
            cal_res["benchmarks"][name] = {
                v: _compute_calibration_free_ap(sc, flat_in, flat_cnt) for v, sc in sc_map.items()
            }
        else:
            k2i_use = k2i
            if name == "codiesp":
                tp = target_paths(name, data_root)
                sc_or, k2i_or = apply_or_groups(sc_map["Step2_deconv"], k2i, tp["groups"])
                sc_map["Step3_compiled"] = sc_or
                k2i_use = k2i_or
            gold_in = [[k2i_use[k] for k in ex.labels if k in k2i_use] for ex in exs]
            gold_cnt = [len(ex.labels) for ex in exs]
            tot_gold = sum(gold_cnt)
            entry = {}
            for v, sc in sc_map.items():
                ap = _compute_calibration_free_ap(sc, gold_in, gold_cnt)
                f1_tau_a = compute_threshold_micro_f1(sc, gold_in, tau_a, total_gold_count=tot_gold)
                f1_unit05 = compute_threshold_micro_f1(sc, gold_in, 0.50, total_gold_count=tot_gold)
                f1_topk = _zero_label_cardinality_f1(sc, gold_in, tot_gold, mean_src_k)
                entry[v] = {
                    **ap,
                    "f1@tau_A": round(f1_tau_a, 4),
                    "f1@unit_0.50": round(f1_unit05, 4),
                    "f1@source_cardinality": round(f1_topk, 4),
                }
            cal_res["benchmarks"][name] = entry
            print(
                f"  [R3:{name}] Step2 mAP={entry['Step2_deconv']['mean_note_ap']:.4f} "
                f"uAUPRC>={entry['Step2_deconv']['micro_auprc_lb']:.4f} "
                f"F1@tauA={entry['Step2_deconv']['f1@tau_A']:.4f} F1@0.50={entry['Step2_deconv']['f1@unit_0.50']:.4f}",
                flush=True,
            )
    return cal_res, drg_versions_res


def run_r4_channel_partitions(
    model: TokenLevelOCSFModel,
    bank: ChannelBank,
    tok: object,
    tau_a: float,
    cache_dir: Path,
    data_root: Path,
    src_subjs: set[int],
    dev: torch.device,
) -> dict:
    """R4: Compare Channel Partitioning Strategies (Spherical k-Means vs Ward vs Random Anchors + Soft Top-2)."""
    print("[R4] Running channel partitioning strategy ablation...", flush=True)
    cb_dir = cache_dir / "pplx-0.6b" / "codebooks"
    z_a = l2_normalize(np.load(cb_dir / "icd9-full.npy").astype(np.float32))
    z_10 = l2_normalize(np.load(cb_dir / "mimiciv-icd10.npy").astype(np.float32))

    rng = np.random.default_rng(7)
    n_ch = bank.n_channels
    src_val = list(iter_source_icd9("validation", data_root, limit=1000))
    src_cb = load_codebook(data_root / "benchmarks/mimiciv-2.2/codebooks/icd9-full.json")
    s_k2i = {e.key: i for i, e in enumerate(src_cb)}
    val_gold_arr = [np.array([s_k2i[k] for k in ex.labels if k in s_k2i], dtype=np.int32) for ex in src_val]
    val_gold_keys = [ex.labels for ex in src_val]

    banks: dict[str, ChannelBank] = {"spherical_kmeans": bank}
    t0 = time.perf_counter()
    ward = AgglomerativeClustering(n_clusters=n_ch, linkage="ward")
    w_lbl = ward.fit_predict(z_a)
    w_cent = np.zeros((n_ch, z_a.shape[1]), dtype=np.float32)
    for r in range(n_ch):
        idx = np.where(w_lbl == r)[0]
        w_cent[r] = z_a[idx].mean(axis=0) if len(idx) else z_a[r]
    banks["hierarchical_ward"] = ChannelBank(anchors=l2_normalize(w_cent))
    print(f"  [R4] Fit Hierarchical Ward R={n_ch} on Z_A in {time.perf_counter()-t0:.1f}s", flush=True)

    cb10 = load_codebook(target_paths("mimiciv-icd10", data_root)["codebook"])
    k2i10 = {e.key: i for i, e in enumerate(cb10)}
    ward_bank = banks["hierarchical_ward"]

    oracle_res: dict = {}
    seeds = (42, 43, 44, 45, 46)
    per_strat: dict[str, list[dict]] = {"spherical_kmeans": [], "hierarchical_ward": [], "random_anchors": []}
    for sd in seeds:
        sd_rng = np.random.default_rng(sd)
        icd10_gold_arr = [sd_rng.choice(len(z_10), size=13, replace=False).astype(np.int32) for _ in range(500)]
        icd10_gold_keys = [tuple(cb10[i].key for i in g) for g in icd10_gold_arr]
        skm_b = ChannelBank.fit(z_a, n_channels=n_ch, seed=sd)
        ridx = sd_rng.choice(len(z_a), size=n_ch, replace=False)
        rnd_b = ChannelBank(anchors=l2_normalize(z_a[ridx].copy()))
        for bname, b_obj in (("spherical_kmeans", skm_b), ("hierarchical_ward", ward_bank), ("random_anchors", rnd_b)):
            nrng = np.random.default_rng(sd + 1000)
            op_a = compile_deconvolution_operator(z_a, b_obj, lam=0.15, self_normalize=True)
            u_a = encode_ground_truth_state(val_gold_arr, z_a, op_a.assign, n_ch)
            u_a_noisy = u_a + nrng.normal(0.0, 0.15, size=u_a.shape).astype(np.float32)
            m_a, _ = evaluate_multilabel(op_a.score_batch(u_a_noisy), val_gold_keys, s_k2i, threshold=0.5)

            op_10 = compile_deconvolution_operator(z_10, b_obj, lam=0.15, self_normalize=True)
            u_10 = encode_ground_truth_state(icd10_gold_arr, z_10, op_10.assign, n_ch)
            u_10_noisy = u_10 + nrng.normal(0.0, 0.15, size=u_10.shape).astype(np.float32)
            m_10, _ = evaluate_multilabel(op_10.score_batch(u_10_noisy), icd10_gold_keys, k2i10, threshold=0.5)
            counts = np.bincount(op_10.assign, minlength=n_ch)
            per_strat[bname].append({
                "seed": sd,
                "icd9_r": float(m_a["r@|y|"]),
                "icd10_r": float(m_10["r@|y|"]),
                "max_nr": int(counts.max()),
                "empty": int(np.sum(counts == 0)),
            })
    for bname, rows in per_strat.items():
        r9 = np.array([x["icd9_r"] for x in rows])
        r10 = np.array([x["icd10_r"] for x in rows])
        mx = np.array([x["max_nr"] for x in rows], dtype=np.float64)
        oracle_res[bname] = {
            "n_seeds": len(rows),
            "icd9_noisy_oracle_r@|y|_mean": round(float(r9.mean()), 4),
            "icd9_noisy_oracle_r@|y|_sd": round(float(r9.std(ddof=1)), 4),
            "icd10_noisy_oracle_r@|y|_mean": round(float(r10.mean()), 4),
            "icd10_noisy_oracle_r@|y|_sd": round(float(r10.std(ddof=1)), 4),
            "icd10_max_channel_mean": round(float(mx.mean()), 1),
            "icd10_max_channel_sd": round(float(mx.std(ddof=1)), 1),
        }
        print(f"  [R4 Oracle 5-seed:{bname:<18}] ICD9={r9.mean():.4f}±{r9.std(ddof=1):.4f} ICD10={r10.mean():.4f}±{r10.std(ddof=1):.4f}", flush=True)

    # Part B: Trained OCSF-E2E zero-retrain target routing: Hard Top-1 vs Soft Top-2 on real ICD-10 (N=2000)
    exs10 = list(iter_target_eval("mimiciv-icd10", data_root, source_subjects=src_subjs, limit=2000))
    g10_keys = [ex.labels for ex in exs10]
    tok10 = _tokenize_texts(tok, [e.text for e in exs10], max_length=3072)

    sims = z_10 @ bank.anchors.T
    top2 = np.argpartition(-sims, 1, axis=1)[:, :2]
    s0 = sims[np.arange(len(z_10)), top2[:, 0]]
    s1 = sims[np.arange(len(z_10)), top2[:, 1]]
    swap = s1 > s0
    a1 = np.where(swap, top2[:, 1], top2[:, 0]).astype(np.int32)
    a2 = np.where(swap, top2[:, 0], top2[:, 1]).astype(np.int32)
    w_top = np.exp(np.stack([np.maximum(s0, s1), np.minimum(s0, s1)], axis=1) / 0.1)
    w_top /= w_top.sum(axis=1, keepdims=True)

    op1 = compile_deconvolution_operator(z_10, bank, lam=0.15, self_normalize=True, assign_override=a1)
    op2 = compile_deconvolution_operator(z_10, bank, lam=0.15, self_normalize=True, assign_override=a2)
    sc_pair = _score_multi_ops(
        model,
        tok10,
        int(tok.pad_token_id),
        dev,
        65536,
        {"top1": StreamingGPUDeconvScorer(op1, dev), "top2": StreamingGPUDeconvScorer(op2, dev)},
    )
    sc_hard = sc_pair["top1"]
    sc_soft2 = sc_pair["top1"] * w_top[:, 0][None, :] + sc_pair["top2"] * w_top[:, 1][None, :]
    sc_max2 = np.maximum(sc_pair["top1"], sc_pair["top2"])
    routing_res = {}
    for rname, sc in [("hard_top1", sc_hard), ("soft_top2_weighted", sc_soft2), ("soft_top2_max", sc_max2)]:
        m, _ = evaluate_multilabel(sc, g10_keys, k2i10, threshold=tau_a)
        routing_res[rname] = {k: round(m[k], 4) for k in ("r@|y|", "micro_f1@tau_A", "p@8", "mrr")}
    print(
        f"  [R4 Routing ICD-10 N=2000] hard_top1={routing_res['hard_top1']['r@|y|']:.4f} "
        f"soft2={routing_res['soft_top2_weighted']['r@|y|']:.4f} max2={routing_res['soft_top2_max']['r@|y|']:.4f}",
        flush=True,
    )
    return {"oracle_partition_capacity": oracle_res, "trained_target_routing_icd10_2k": routing_res}


@torch.inference_mode()
def run_r5_directionality_and_length_diag(
    run_dir_pplx: Path,
    run_dir_qwen: Path,
    cache_dir: Path,
    data_root: Path,
    src_subjs: set[int],
    dev: torch.device,
) -> dict:
    """R5a: Exact same-weight is_causal=False vs True intervention on frozen pplx-0.6b.
    R5b: Note-length tercile stratification of pplx-0.6b vs qwen-0.6b on ICD-9 & ICD-10."""
    print("[R5a] Running exact same-weight is_causal intervention on frozen pplx-0.6b...", flush=True)
    spec = BACKBONE_SPECS["pplx-0.6b"]
    tok = AutoTokenizer.from_pretrained(spec.hf_id, trust_remote_code=spec.trust_remote_code)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    raw_bb = AutoModel.from_pretrained(
        spec.hf_id,
        trust_remote_code=spec.trust_remote_code,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to(dev)
    raw_bb.eval()

    def _set_causal_flag(module: torch.nn.Module, flag: bool) -> int:
        n_set = 0
        for m in module.modules():
            if hasattr(m, "is_causal"):
                m.is_causal = flag
                n_set += 1
        return n_set

    cb_dir = cache_dir / "pplx-0.6b" / "codebooks"
    causal_diag: dict = {}
    for tgt, limit in (("mimiciv-icd10", 1500), ("elcardiocc-2026", 1500)):
        tp = target_paths(tgt, data_root)
        cb = load_codebook(tp["codebook"])
        k2i = {e.key: i for i, e in enumerate(cb)}
        exs = list(iter_target_eval(tgt, data_root, source_subjects=src_subjs, limit=limit))
        z_t = l2_normalize(np.load(cb_dir / f"{tgt}.npy").astype(np.float32))
        toks = _tokenize_texts(tok, [e.text for e in exs], max_length=3072)
        batches = make_length_bucketed_batches([len(s) for s in toks], max_tokens=65536, max_batch_size=32)

        tgt_d = {}
        for is_c in (False, True):
            n_toggled = _set_causal_flag(raw_bb, is_c)
            embs = np.empty((len(exs), z_t.shape[1]), dtype=np.float32)
            for b in batches:
                ids_t, mask_t = _collate_ids(toks, b, int(tok.pad_token_id), dev)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    h = torch.tanh(raw_bb(input_ids=ids_t, attention_mask=mask_t, use_cache=False).last_hidden_state)
                mf = mask_t.unsqueeze(-1).float()
                doc = F.normalize((h.float() * mf).sum(dim=1) / mf.sum(dim=1).clamp(min=1.0), p=2, dim=-1)
                arr = doc.cpu().numpy()
                for k, idx in enumerate(b):
                    embs[idx] = arr[k]
            scores = embs @ z_t.T
            if tgt == "elcardiocc-2026":
                m, _ = evaluate_answer_groups(scores, [ex.label_groups for ex in exs], k2i)
                tgt_d["causal_forced" if is_c else "bidirectional_native"] = {
                    "layers_toggled": n_toggled,
                    "acc@1": round(m["acc@1"], 4),
                    "acc@5": round(m["acc@5"], 4),
                    "mrr": round(m["mrr"], 4),
                }
            else:
                m, _ = evaluate_multilabel(scores, [ex.labels for ex in exs], k2i, threshold=0.2)
                tgt_d["causal_forced" if is_c else "bidirectional_native"] = {
                    "layers_toggled": n_toggled,
                    "r@|y|": round(m["r@|y|"], 4),
                    "p@8": round(m["p@8"], 4),
                    "mrr": round(m["mrr"], 4),
                }
        causal_diag[tgt] = tgt_d
        print(f"  [R5a:{tgt}] {tgt_d}", flush=True)
    del raw_bb
    torch.cuda.empty_cache()

    print("[R5b] Computing note-length tercile margin (pplx-0.6b minus qwen-0.6b)...", flush=True)
    len_diag: dict = {}
    for bench, exs in [
        ("source_icd9_val", list(iter_source_icd9("validation", data_root))),
        ("mimiciv-icd10", list(iter_target_eval("mimiciv-icd10", data_root, source_subjects=src_subjs))),
    ]:
        p_arr = np.load(run_dir_pplx / "per_note" / "Step2_OCSF_deconv" / f"{bench}.npy")
        q_arr = np.load(run_dir_qwen / "per_note" / "Step2_OCSF_deconv" / f"{bench}.npy")
        valid_exs = [e for e in exs if len(e.labels) > 0]
        if len(valid_exs) != len(p_arr):
            valid_exs = exs[: len(p_arr)]
        w_lens = np.array([len(e.text.split()) for e in valid_exs], dtype=np.int32)
        q33, q67 = np.quantile(w_lens, [1.0 / 3.0, 2.0 / 3.0])
        terciles = {}
        for tname, cond in [
            (f"short (<={int(q33)}w)", w_lens <= q33),
            (f"medium ({int(q33)+1}-{int(q67)}w)", (w_lens > q33) & (w_lens <= q67)),
            (f"long (>{int(q67)}w)", w_lens > q67),
        ]:
            idx = np.where(cond)[0]
            terciles[tname] = {
                "n_notes": int(idx.size),
                "pplx_mean": round(float(p_arr[idx].mean()), 4),
                "qwen_mean": round(float(q_arr[idx].mean()), 4),
                "delta": round(float((p_arr[idx] - q_arr[idx]).mean()), 4),
            }
        len_diag[bench] = terciles
        print(f"  [R5b:{bench}] {terciles}", flush=True)

    return {"same_weight_causal_intervention": causal_diag, "length_terciles_pplx_minus_qwen": len_diag}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir-pplx06", type=Path, required=True)
    parser.add_argument("--run-dir-qwen06", type=Path, required=True)
    parser.add_argument("--run-dir-pplx4b", type=Path, default=None)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--out-json", type=Path, required=True)
    args = parser.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    src_subjs = load_source_subject_ids(args.data_root)

    r1 = run_r1_latency_vram_profile(args.run_dir_pplx06, args.run_dir_pplx4b, args.cache_dir, dev)

    model06, bank06, tok06, tau_a = _load_ocsf_model("pplx-0.6b", args.run_dir_pplx06, dev)
    r2 = run_r2_robustness(model06, bank06, tok06, tau_a, args.cache_dir, args.data_root, src_subjs, dev)
    r3, r5c = run_r3_calibration_and_r5_drg_versions(
        model06, bank06, tok06, tau_a, args.cache_dir, args.data_root, src_subjs, dev
    )
    r4 = run_r4_channel_partitions(model06, bank06, tok06, tau_a, args.cache_dir, args.data_root, src_subjs, dev)
    del model06
    torch.cuda.empty_cache()

    r5 = run_r5_directionality_and_length_diag(
        args.run_dir_pplx06, args.run_dir_qwen06, args.cache_dir, args.data_root, src_subjs, dev
    )
    r5["ms_drg_per_cms_version"] = r5c

    payload = {
        "R1_latency_vram_profile": r1,
        "R2_description_noise_robustness": r2,
        "R3_calibration_free_and_plugin": r3,
        "R4_channel_partition_ablation": r4,
        "R5_directionality_length_drg_versions": r5,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[reviewer-suite] ALL DONE -> {args.out_json}", flush=True)


if __name__ == "__main__":
    main()
