"""One-shot locked evaluation on the four official sealed test splits (user-approved 2026-11-22).

Strictly read-only and zero-tuning:
  - Loads pre-trained, frozen checkpoints, ChannelBanks, and source-validation thresholds tau_A.
  - Computes source_mean_state strictly on source ICD-9 validation (never on test).
  - Evaluates on:
      1. source_icd9_test           (32,804 notes, 15,425 patients; Full 11,324 + Edin Freq-6,150)
      2. mimiciv-icd10_test         (13,783 patient-disjoint notes; All + CM + PCS + Freq + Rare)
      3. mimiciv-ms-drg_test        (11,869 patient-disjoint notes; answer-group acc@1/acc@5/MRR)
      4. codiesp_test               (250 Spanish notes; 185,458 codes + 11,381 OR groups)
  - Saves per-note metric arrays for exact sealed-test paired bootstrap significance testing.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import gzip
import json
from pathlib import Path
import time
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from ocsf.compiler import apply_or_groups, compile_drg_operator, DRGCompositionalOperator
from ocsf.data import (
    DEFAULT_DATA_ROOT,
    iter_source_icd9,
    load_codebook,
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
    evaluate_answer_groups,
    evaluate_multilabel,
    paired_bootstrap_ci,
)
from ocsf.state_matrix import ChannelBank, DeconvolutionOperator, compile_deconvolution_operator, l2_normalize


def _load_key_subset(path: Path) -> set[str]:
    return {e.key for e in load_codebook(path)}


def _load_jsonl_gz(path: Path) -> list[dict]:
    rows: list[dict] = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _tokenize_texts(tok, texts: list[str], max_length: int) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    for i in range(0, len(texts), 512):
        enc = tok(texts[i : i + 512], add_special_tokens=True, truncation=True, max_length=max_length)
        for ids in enc["input_ids"]:
            out.append(np.asarray(ids, dtype=np.int32))
    return out


def _collate(tokens: list[np.ndarray], idxs: list[int], pad_id: int, dev: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    max_l = max(len(tokens[i]) for i in idxs)
    max_l = ((max_l + 7) // 8) * 8
    bsz = len(idxs)
    ids = torch.full((bsz, max_l), pad_id, dtype=torch.long)
    mask = torch.zeros((bsz, max_l), dtype=torch.long)
    for row, idx in enumerate(idxs):
        arr = tokens[idx]
        l = len(arr)
        ids[row, :l] = torch.from_numpy(arr.astype(np.int64))
        mask[row, :l] = 1
    return ids.to(dev, non_blocking=True), mask.to(dev, non_blocking=True)


def _dot_op(z: np.ndarray, bank: ChannelBank) -> DeconvolutionOperator:
    zu = l2_normalize(z.astype(np.float32, copy=False))
    assign = bank.assign_hard(zu)
    c_idx, c_w = [], []
    for r in range(bank.n_channels):
        idx = np.where(assign == r)[0].astype(np.int32)
        c_idx.append(idx)
        c_w.append(zu[idx] if idx.size else np.zeros((0, zu.shape[1]), dtype=np.float32))
    return DeconvolutionOperator(
        n_codes=zu.shape[0],
        n_channels=bank.n_channels,
        dim=zu.shape[1],
        assign=assign,
        chan_indices=tuple(c_idx),
        chan_weights=tuple(c_w),
        diag_gains=np.ones(zu.shape[0], dtype=np.float32),
    )


def _compute_mAP_and_auprc_lb(scores: np.ndarray, gold_in: list[list[int]], gold_cnt: list[int], top_k: int = 200) -> dict[str, float]:
    n, c = scores.shape
    k = min(top_k, c)
    aps: list[float] = []
    all_s: list[np.ndarray] = []
    all_y: list[np.ndarray] = []
    tot_pos = 0
    for i in range(n):
        g_tot = gold_cnt[i]
        if g_tot <= 0:
            continue
        tot_pos += g_tot
        gset = set(gold_in[i])
        row = scores[i]
        idx = np.argpartition(-row, k - 1)[:k]
        idx = idx[np.argsort(-row[idx])]
        s_top = row[idx].astype(np.float32)
        y_top = np.fromiter((1 if int(j) in gset else 0 for j in idx), dtype=np.int8, count=k)
        all_s.append(s_top)
        all_y.append(y_top)
        hits = np.where(y_top == 1)[0]
        if hits.size == 0:
            aps.append(0.0)
        else:
            prec = np.arange(1, hits.size + 1, dtype=np.float64) / (hits + 1.0)
            aps.append(float(prec.sum() / g_tot))
    cat_s = np.concatenate(all_s)
    cat_y = np.concatenate(all_y)
    order = np.argsort(-cat_s, kind="mergesort")
    ys = cat_y[order]
    tp = np.cumsum(ys, dtype=np.float64)
    prec_g = tp / np.arange(1, ys.size + 1, dtype=np.float64)
    auprc_lb = float(prec_g[ys == 1].sum() / max(tot_pos, 1))
    return {"mean_note_ap": float(np.mean(aps)), "micro_auprc_lb": auprc_lb}


@torch.inference_mode()
def _score_ocsf_variants(
    model: TokenLevelOCSFModel,
    tokens: list[np.ndarray],
    pad_id: int,
    dev: torch.device,
    max_eval_tokens: int,
    ops: dict[str, DeconvolutionOperator | DRGCompositionalOperator],
    source_mean_state: torch.Tensor | None = None,
    debias_variants: set[str] | None = None,
    debias_gamma: float = 0.5,
) -> dict[str, np.ndarray]:
    n = len(tokens)
    batches = make_length_bucketed_batches([len(s) for s in tokens], max_tokens=max_eval_tokens, max_batch_size=32)
    debias_variants = debias_variants or set()

    gpu_ops: dict[str, object] = {}
    out_scores: dict[str, np.ndarray] = {}
    for vname, op in ops.items():
        if isinstance(op, DRGCompositionalOperator):
            gpu_ops[vname] = (
                StreamingGPUDeconvScorer(op.direct_op, dev),
                StreamingGPUDeconvScorer(op.prototype_op, dev),
                op,
            )
            out_scores[vname] = np.empty((n, op.direct_op.n_codes), dtype=np.float32)
        else:
            gpu_ops[vname] = StreamingGPUDeconvScorer(op, dev)
            out_scores[vname] = np.empty((n, op.n_codes), dtype=np.float32)

    mean_dev = source_mean_state.to(dev) if source_mean_state is not None else None
    for b in batches:
        ids_t, mask_t = _collate(tokens, b, pad_id, dev)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            u = model(ids_t, mask_t).float()
        u_deb = (u - debias_gamma * mean_dev.unsqueeze(0)) if mean_dev is not None else u
        for vname, gop in gpu_ops.items():
            u_use = u_deb if vname in debias_variants else u
            if isinstance(gop, tuple):
                dir_sc, proto_sc, drg_op = gop
                s_dir = dir_sc.score_tensor(u_use)
                s_pro = proto_sc.score_tensor(u_use)
                arr = drg_op.combine_scores(s_dir, s_pro)
            else:
                arr = gop.score_tensor(u_use)
            for k, idx in enumerate(b):
                out_scores[vname][idx] = arr[k]
    return out_scores


@torch.inference_mode()
def _compute_source_val_mean_state(
    model: TokenLevelOCSFModel,
    tok,
    data_root: Path,
    dev: torch.device,
    max_length: int,
    max_eval_tokens: int,
) -> torch.Tensor:
    """Strictly non-test source-validation mean state bar_U_A (never touches test!)."""
    val_exs = list(iter_source_icd9("validation", data_root))
    toks = _tokenize_texts(tok, [e.text for e in val_exs], max_length=max_length)
    batches = make_length_bucketed_batches([len(s) for s in toks], max_tokens=max_eval_tokens, max_batch_size=32)
    acc = torch.zeros((model.decoder.n_channels, model.decoder.dim), device=dev, dtype=torch.float64)
    total = 0
    for b in batches:
        ids_t, mask_t = _collate(toks, b, int(tok.pad_token_id), dev)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            u = model(ids_t, mask_t).float()
        acc += u.double().sum(dim=0)
        total += len(b)
    return (acc / max(total, 1)).float().cpu()


def eval_ocsf_checkpoint(
    tag: str,
    backbone: str,
    run_dir: Path,
    test_dir: Path,
    cache_dir: Path,
    data_root: Path,
    out_dir: Path,
    dev: torch.device,
    include_unnorm: bool = False,
    max_eval_tokens: int = 65536,
) -> dict:
    t_start = time.perf_counter()
    print(f"\n[sealed-test] === Evaluating OCSF checkpoint: {tag} ({backbone}) ===", flush=True)
    ckpt = torch.load(run_dir / "best_e2e_trainable.pt", map_location="cpu", weights_only=False)
    cfg = ckpt.get("config", {})
    tau_a = float(ckpt["tau_A"] if "tau_A" in ckpt else ckpt["tau_a"])
    lam = float(cfg.get("lam", 0.15))
    max_len = int(cfg.get("max_tokens", 3072))
    lora_r = int(cfg.get("lora_r", 32))
    lora_alpha = float(cfg.get("lora_alpha", 64.0))
    has_bb_layer0 = any(k.startswith("backbone.") and ".layers.0." in k for k in ckpt["trainable_state"])
    freeze_ratio = 0.0 if has_bb_layer0 else float(cfg.get("freeze_ratio", 0.5))
    debias_gamma = float(cfg.get("debias_gamma", 0.5))

    bank = ChannelBank(anchors=np.load(run_dir / "channel_bank.npz")["anchors"].astype(np.float32))
    spec = BACKBONE_SPECS[backbone]
    tok = AutoTokenizer.from_pretrained(spec.hf_id, trust_remote_code=spec.trust_remote_code)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    pad_id = int(tok.pad_token_id)

    bb = AutoModel.from_pretrained(
        spec.hf_id,
        trust_remote_code=spec.trust_remote_code,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    inject_upper_layer_lora(bb, r=lora_r, alpha=lora_alpha, dropout=0.0, freeze_ratio=freeze_ratio)
    dec = TokenChannelStateDecoder(torch.from_numpy(bank.anchors), n_heads=8, n_channel_layers=2, dropout=0.0)
    model = TokenLevelOCSFModel(bb, dec, apply_tanh=(spec.family == "pplx_diffusion"))
    missing, unexpected = model.load_state_dict(ckpt["trainable_state"], strict=False)
    assert not unexpected, f"Unexpected keys in {tag}: {unexpected[:5]}"
    model.to(dev).eval()
    print(f"  Loaded ep={ckpt.get('epoch')} tau_A={tau_a:.4f} R={bank.n_channels} dim={bank.anchors.shape[1]} frz={freeze_ratio}", flush=True)

    # Compute frozen source_mean_state strictly from source ICD-9 validation (non-test!)
    t_ms = time.perf_counter()
    source_mean_state = _compute_source_val_mean_state(model, tok, data_root, dev, max_len, max_eval_tokens)
    print(f"  Computed non-test source_val mean_state in {time.perf_counter()-t_ms:.1f}s", flush=True)

    cb_dir = cache_dir / backbone / "codebooks"
    pn_dir = out_dir / "per_note" / tag
    pn_dir.mkdir(parents=True, exist_ok=True)
    res: dict[str, dict] = {
        "meta": {
            "tag": tag,
            "backbone": backbone,
            "run_dir": str(run_dir),
            "epoch": int(ckpt.get("epoch", -1)),
            "tau_a": tau_a,
            "lam": lam,
            "n_channels": int(bank.n_channels),
            "dim": int(bank.anchors.shape[1]),
            "freeze_ratio": freeze_ratio,
        },
        "results": {},
        "bootstrap": {},
    }

    # --- 1. Official Source ICD-9 Test (32,804 notes; Full 11,324 + Edin Freq-6,150) ---
    rows9 = _load_jsonl_gz(test_dir / "source_icd9_test.jsonl.gz")
    cb9 = load_codebook(data_root / "benchmarks/mimiciv-2.2/codebooks/icd9-full.json")
    k2i9 = {e.key: i for i, e in enumerate(cb9)}
    freq9_keys = _load_key_subset(data_root / "benchmarks/mimiciv-2.2/codebooks/icd9-frequent.json")
    freq9_idx = np.array([k2i9[k] for k in freq9_keys if k in k2i9], dtype=np.int32)
    freq9_k2i = {k: i for i, k in enumerate([cb9[j].key for j in freq9_idx])}
    z9 = l2_normalize(np.load(cb_dir / "icd9-full.npy").astype(np.float32))
    ops9 = {
        "B3_2D_dot": _dot_op(z9, bank),
        "Step2_OCSF_deconv": compile_deconvolution_operator(z9, bank, lam=lam, self_normalize=True),
    }
    if include_unnorm:
        ops9["Step2_unnorm_deconv"] = compile_deconvolution_operator(z9, bank, lam=lam, self_normalize=False)

    toks9 = _tokenize_texts(tok, [r["text"] for r in rows9], max_length=max_len)
    sc9 = _score_ocsf_variants(model, toks9, pad_id, dev, max_eval_tokens, ops9)
    gold9 = [tuple(r["labels"]) for r in rows9]
    gold9_freq = [tuple(k for k in g if k in freq9_k2i) for g in gold9]
    gold9_in = [[k2i9[k] for k in g if k in k2i9] for g in gold9]
    gold9_cnt = [len(g) for g in gold9]
    for v, sc in sc9.items():
        m_full, pn_full = evaluate_multilabel(sc, gold9, k2i9, threshold=tau_a)
        m_freq, pn_freq = evaluate_multilabel(sc[:, freq9_idx], gold9_freq, freq9_k2i, threshold=tau_a)
        ap_full = _compute_mAP_and_auprc_lb(sc, gold9_in, gold9_cnt)
        m_full.update(ap_full)
        res["results"].setdefault(v, {})["source_icd9_test"] = m_full
        res["results"][v]["source_icd9_test_frequent"] = m_freq
        np.save(pn_dir / f"{v}__source_icd9_test.npy", pn_full)
        np.save(pn_dir / f"{v}__source_icd9_test_frequent.npy", pn_freq)
    print(
        f"  [ICD-9 Test N={len(rows9):,}] Step2 Full F1={res['results']['Step2_OCSF_deconv']['source_icd9_test']['micro_f1@tau_A']:.4f} "
        f"R@|y|={res['results']['Step2_OCSF_deconv']['source_icd9_test']['r@|y|']:.4f} "
        f"P@15={res['results']['Step2_OCSF_deconv']['source_icd9_test']['p@15']:.4f} | "
        f"Freq6150 F1={res['results']['Step2_OCSF_deconv']['source_icd9_test_frequent']['micro_f1@tau_A']:.4f} "
        f"R@|y|={res['results']['Step2_OCSF_deconv']['source_icd9_test_frequent']['r@|y|']:.4f} "
        f"P@8={res['results']['Step2_OCSF_deconv']['source_icd9_test_frequent']['p@8']:.4f}",
        flush=True,
    )
    del sc9, toks9, rows9

    # --- 2. Official MIMIC-IV ICD-10 Patient-Disjoint Test (13,783 notes; All + CM + PCS + Freq + Rare) ---
    rows10 = _load_jsonl_gz(test_dir / "icd10_test.jsonl.gz")
    tp10 = target_paths("mimiciv-icd10", data_root)
    cb10 = load_codebook(tp10["codebook"])
    k2i10 = {e.key: i for i, e in enumerate(cb10)}
    freq10_keys = _load_key_subset(tp10["frequent_codebook"])
    z10 = l2_normalize(np.load(cb_dir / "mimiciv-icd10.npy").astype(np.float32))
    ops10 = {
        "B3_2D_dot": _dot_op(z10, bank),
        "Step2_OCSF_deconv": compile_deconvolution_operator(z10, bank, lam=lam, self_normalize=True),
    }
    if include_unnorm:
        ops10["Step2_unnorm_deconv"] = compile_deconvolution_operator(z10, bank, lam=lam, self_normalize=False)

    toks10 = _tokenize_texts(tok, [r["text"] for r in rows10], max_length=max_len)
    sc10 = _score_ocsf_variants(model, toks10, pad_id, dev, max_eval_tokens, ops10)
    gold10 = [tuple(r["labels"]) for r in rows10]
    slices10 = {
        "mimiciv-icd10_test": np.arange(len(cb10), dtype=np.int32),
        "mimiciv-icd10_test_cm": np.array([i for i, e in enumerate(cb10) if e.system == "icd10_diag"], dtype=np.int32),
        "mimiciv-icd10_test_pcs": np.array([i for i, e in enumerate(cb10) if e.system == "icd10_proc"], dtype=np.int32),
        "mimiciv-icd10_test_frequent": np.array([i for i, e in enumerate(cb10) if e.key in freq10_keys], dtype=np.int32),
        "mimiciv-icd10_test_rare": np.array([i for i, e in enumerate(cb10) if e.key not in freq10_keys], dtype=np.int32),
    }
    gold10_in = [[k2i10[k] for k in g if k in k2i10] for g in gold10]
    gold10_cnt = [len(g) for g in gold10]
    for v, sc in sc10.items():
        for sname, sidx in slices10.items():
            sk2i = {cb10[j].key: i for i, j in enumerate(sidx)}
            sgold = [tuple(k for k in g if k in sk2i) for g in gold10]
            m, pn = evaluate_multilabel(sc[:, sidx], sgold, sk2i, threshold=tau_a)
            if sname == "mimiciv-icd10_test":
                m.update(_compute_mAP_and_auprc_lb(sc, gold10_in, gold10_cnt))
            res["results"][v][sname] = m
            np.save(pn_dir / f"{v}__{sname}.npy", pn)
    m10 = res["results"]["Step2_OCSF_deconv"]["mimiciv-icd10_test"]
    m10cm = res["results"]["Step2_OCSF_deconv"]["mimiciv-icd10_test_cm"]
    m10pcs = res["results"]["Step2_OCSF_deconv"]["mimiciv-icd10_test_pcs"]
    print(
        f"  [ICD-10 Test N={len(rows10):,}] All R@|y|={m10['r@|y|']:.4f} F1={m10['micro_f1@tau_A']:.4f} P@8={m10['p@8']:.4f} "
        f"mAP={m10['mean_note_ap']:.4f} | CM={m10cm['r@|y|']:.4f} (F1={m10cm['micro_f1@tau_A']:.4f}) | PCS={m10pcs['r@|y|']:.4f}",
        flush=True,
    )
    del sc10, toks10, rows10

    # --- 3. Official MIMIC-IV MS-DRG Patient-Disjoint Test (11,869 notes) ---
    rows_drg = _load_jsonl_gz(test_dir / "drg_test.jsonl.gz")
    tp_drg = target_paths("mimiciv-ms-drg", data_root)
    cb_drg = load_codebook(tp_drg["codebook"])
    k2i_drg = {e.key: i for i, e in enumerate(cb_drg)}
    z_drg = l2_normalize(np.load(cb_dir / "mimiciv-ms-drg.npy").astype(np.float32))
    proto_npz = np.load(cb_dir / "drg_prototypes.npz")
    drg_protos = {k: proto_npz[k] for k in proto_npz.files}
    ops_drg = {
        "B3_2D_dot": _dot_op(z_drg, bank),
        "Step2_OCSF_deconv": compile_deconvolution_operator(z_drg, bank, lam=lam, self_normalize=True),
        "Step3_OCSF_compiled": compile_drg_operator(
            cb_drg, z_drg, drg_protos, bank, cms_dir=tp_drg["cms_dir"], alignment=None, lam=lam
        ),
    }
    if include_unnorm:
        ops_drg["Step2_unnorm_deconv"] = compile_deconvolution_operator(z_drg, bank, lam=lam, self_normalize=False)

    toks_drg = _tokenize_texts(tok, [r["text"] for r in rows_drg], max_length=max_len)
    sc_drg = _score_ocsf_variants(
        model, toks_drg, pad_id, dev, max_eval_tokens, ops_drg,
        source_mean_state=source_mean_state,
        debias_variants={"Step3_OCSF_compiled"},
        debias_gamma=debias_gamma,
    )
    grps_drg = [tuple(tuple(g) for g in r["label_groups"]) for r in rows_drg]
    for v, sc in sc_drg.items():
        m, pn = evaluate_answer_groups(sc, grps_drg, k2i_drg)
        res["results"].setdefault(v, {})["mimiciv-ms-drg_test"] = m
        np.save(pn_dir / f"{v}__mimiciv-ms-drg_test.npy", pn)
    mdrg = res["results"]["Step3_OCSF_compiled"]["mimiciv-ms-drg_test"]
    print(f"  [MS-DRG Test N={len(rows_drg):,}] Step3 acc@1={mdrg['acc@1']:.4f} acc@5={mdrg['acc@5']:.4f} MRR={mdrg['mrr']:.4f}", flush=True)
    del sc_drg, toks_drg, rows_drg

    # --- 4. Official CodiEsp Spanish Test (250 notes, 185,458 codes + 11,381 OR groups) ---
    rows_cod = _load_jsonl_gz(test_dir / "codiesp_test.jsonl.gz")
    tp_cod = target_paths("codiesp", data_root)
    cb_cod = load_codebook(tp_cod["codebook"])
    k2i_cod = {e.key: i for i, e in enumerate(cb_cod)}
    z_cod = l2_normalize(np.load(cb_dir / "codiesp.npy").astype(np.float32))
    ops_cod = {
        "B3_2D_dot": _dot_op(z_cod, bank),
        "Step2_OCSF_deconv": compile_deconvolution_operator(z_cod, bank, lam=lam, self_normalize=True),
    }
    if include_unnorm:
        ops_cod["Step2_unnorm_deconv"] = compile_deconvolution_operator(z_cod, bank, lam=lam, self_normalize=False)

    toks_cod = _tokenize_texts(tok, [r["text"] for r in rows_cod], max_length=max_len)
    sc_cod = _score_ocsf_variants(model, toks_cod, pad_id, dev, max_eval_tokens, ops_cod)
    sc_cod_or, k2i_cod_or = apply_or_groups(sc_cod["Step2_OCSF_deconv"], k2i_cod, tp_cod["groups"])
    sc_cod["Step3_OCSF_compiled"] = sc_cod_or
    gold_cod = [tuple(r["labels"]) for r in rows_cod]
    for v, sc in sc_cod.items():
        km = k2i_cod_or if v == "Step3_OCSF_compiled" else k2i_cod
        m, pn = evaluate_multilabel(sc, gold_cod, km, threshold=tau_a)
        g_in = [[km[k] for k in g if k in km] for g in gold_cod]
        m.update(_compute_mAP_and_auprc_lb(sc, g_in, [len(g) for g in gold_cod]))
        res["results"].setdefault(v, {})["codiesp_test"] = m
        np.save(pn_dir / f"{v}__codiesp_test.npy", pn)
    mcod = res["results"]["Step3_OCSF_compiled"]["codiesp_test"]
    print(f"  [CodiEsp Test N={len(rows_cod):,}] Step3 R@|y|={mcod['r@|y|']:.4f} F1={mcod['micro_f1@tau_A']:.4f} P@8={mcod['p@8']:.4f} MRR={mcod['mrr']:.4f}", flush=True)

    # Internal paired bootstrap CIs on sealed test
    for comp_name, v_base, v_cand in [
        ("Step2_minus_B3_dot", "B3_2D_dot", "Step2_OCSF_deconv"),
        ("Step3_minus_Step2", "Step2_OCSF_deconv", "Step3_OCSF_compiled"),
        ("SelfNorm_minus_Unnorm", "Step2_unnorm_deconv", "Step2_OCSF_deconv"),
    ]:
        comp_d = {}
        for f in sorted(pn_dir.glob(f"{v_cand}__*.npy")):
            bench = f.stem.split("__", 1)[1]
            fb = pn_dir / f"{v_base}__{bench}.npy"
            if fb.exists():
                comp_d[bench] = asdict(paired_bootstrap_ci(np.load(fb), np.load(f)))
        if comp_d:
            res["bootstrap"][comp_name] = comp_d

    res["meta"]["elapsed_sec"] = round(time.perf_counter() - t_start, 1)
    out_path = out_dir / f"sealed_test_{tag}.json"
    out_path.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"[sealed-test] Saved {out_path} in {res['meta']['elapsed_sec']}s", flush=True)
    del model, bb, dec
    torch.cuda.empty_cache()
    return res


@torch.inference_mode()
def eval_closed_laat(
    run_dir: Path,
    test_dir: Path,
    data_root: Path,
    out_dir: Path,
    dev: torch.device,
    max_eval_tokens: int = 65536,
) -> dict:
    t0 = time.perf_counter()
    print("\n[sealed-test] === Evaluating Closed-Vocab LAAT (pplx-0.6b) on source_icd9_test ===", flush=True)
    ckpt = torch.load(run_dir / "best_e2e_trainable.pt", map_location="cpu", weights_only=False)
    cfg = ckpt.get("config", {})
    tau_a = float(ckpt["tau_A"] if "tau_A" in ckpt else ckpt["tau_a"])
    max_len = int(cfg.get("max_tokens", 3072))

    spec = BACKBONE_SPECS["pplx-0.6b"]
    tok = AutoTokenizer.from_pretrained(spec.hf_id, trust_remote_code=spec.trust_remote_code)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    pad_id = int(tok.pad_token_id)

    bb = AutoModel.from_pretrained(spec.hf_id, trust_remote_code=spec.trust_remote_code, dtype=torch.bfloat16, attn_implementation="sdpa")
    inject_upper_layer_lora(bb, r=int(cfg.get("lora_r", 32)), alpha=float(cfg.get("lora_alpha", 64.0)), dropout=0.0, freeze_ratio=float(cfg.get("freeze_ratio", 0.5)))

    cb9 = load_codebook(data_root / "benchmarks/mimiciv-2.2/codebooks/icd9-full.json")
    k2i9 = {e.key: i for i, e in enumerate(cb9)}
    freq9_keys = _load_key_subset(data_root / "benchmarks/mimiciv-2.2/codebooks/icd9-frequent.json")
    freq9_idx = np.array([k2i9[k] for k in freq9_keys if k in k2i9], dtype=np.int32)
    freq9_k2i = {cb9[j].key: i for i, j in enumerate(freq9_idx)}

    class _ClosedWrapper(torch.nn.Module):
        def __init__(self, b, h):
            super().__init__()
            self.backbone = b
            self.head = h

        def forward(self, ids, mask):
            out = self.backbone(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state
            h = F.normalize(torch.tanh(out).float(), p=2, dim=-1).to(out.dtype)
            return self.head(h, mask)

    head = ClosedVocabLAATHead(dim=1024, n_codes=len(cb9), proj_dim=512, dropout=0.0)
    wrap = _ClosedWrapper(bb, head)
    missing, unexpected = wrap.load_state_dict(ckpt["trainable_state"], strict=False)
    assert not unexpected, f"Unexpected keys in closed-laat: {unexpected[:5]}"
    wrap.to(dev).eval()

    rows9 = _load_jsonl_gz(test_dir / "source_icd9_test.jsonl.gz")
    toks9 = _tokenize_texts(tok, [r["text"] for r in rows9], max_length=max_len)
    batches = make_length_bucketed_batches([len(s) for s in toks9], max_tokens=max_eval_tokens, max_batch_size=32)
    logits_all = np.empty((len(rows9), len(cb9)), dtype=np.float32)
    for b in batches:
        ids_t, mask_t = _collate(toks9, b, pad_id, dev)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            lg = wrap(ids_t, mask_t).float().cpu().numpy()
        for k, idx in enumerate(b):
            logits_all[idx] = lg[k]

    gold9 = [tuple(r["labels"]) for r in rows9]
    gold9_freq = [tuple(k for k in g if k in freq9_k2i) for g in gold9]
    m_full, pn_full = evaluate_multilabel(logits_all, gold9, k2i9, threshold=tau_a)
    m_freq, pn_freq = evaluate_multilabel(logits_all[:, freq9_idx], gold9_freq, freq9_k2i, threshold=tau_a)
    m_full.update(_compute_mAP_and_auprc_lb(logits_all, [[k2i9[k] for k in g if k in k2i9] for g in gold9], [len(g) for g in gold9]))

    pn_dir = out_dir / "per_note" / "closed_laat_pplx06"
    pn_dir.mkdir(parents=True, exist_ok=True)
    np.save(pn_dir / "closed__source_icd9_test.npy", pn_full)
    np.save(pn_dir / "closed__source_icd9_test_frequent.npy", pn_freq)

    res = {
        "meta": {"tag": "closed_laat_pplx06", "backbone": "pplx-0.6b", "tau_a": tau_a, "epoch": int(ckpt.get("epoch", -1)), "elapsed_sec": round(time.perf_counter() - t0, 1)},
        "source_icd9_test": m_full,
        "source_icd9_test_frequent": m_freq,
    }
    out_path = out_dir / "sealed_test_closed_laat_pplx06.json"
    out_path.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(
        f"  [Closed-LAAT ICD-9 Test] Full F1={m_full['micro_f1@tau_A']:.4f} R@|y|={m_full['r@|y|']:.4f} P@15={m_full['p@15']:.4f} | "
        f"Freq6150 F1={m_freq['micro_f1@tau_A']:.4f} R@|y|={m_freq['r@|y|']:.4f} P@8={m_freq['p@8']:.4f}",
        flush=True,
    )
    del wrap, bb, head
    torch.cuda.empty_cache()
    return res


@torch.inference_mode()
def eval_b0_doc(
    backbone: str,
    test_dir: Path,
    cache_dir: Path,
    data_root: Path,
    out_dir: Path,
    dev: torch.device,
    tau_a_override: float,
    max_eval_tokens: int = 65536,
) -> dict:
    t0 = time.perf_counter()
    print(f"\n[sealed-test] === Evaluating Zero-Shot B0_doc ({backbone}, frozen tau_A={tau_a_override:.4f}) ===", flush=True)
    spec = BACKBONE_SPECS[backbone]
    tok = AutoTokenizer.from_pretrained(spec.hf_id, trust_remote_code=spec.trust_remote_code)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    pad_id = int(tok.pad_token_id)
    bb = AutoModel.from_pretrained(spec.hf_id, trust_remote_code=spec.trust_remote_code, dtype=torch.bfloat16, attn_implementation="sdpa").to(dev).eval()
    cb_dir = cache_dir / backbone / "codebooks"

    def _encode_doc(texts: list[str]) -> np.ndarray:
        toks = _tokenize_texts(tok, texts, max_length=3072)
        batches = make_length_bucketed_batches([len(s) for s in toks], max_tokens=max_eval_tokens, max_batch_size=32)
        dim = int(bb.config.hidden_size)
        out = np.empty((len(texts), dim), dtype=np.float32)
        for b in batches:
            ids_t, mask_t = _collate(toks, b, pad_id, dev)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                h = bb(input_ids=ids_t, attention_mask=mask_t, use_cache=False).last_hidden_state
                if spec.family == "pplx_diffusion":
                    h = torch.tanh(h)
                    mf = mask_t.unsqueeze(-1).float()
                    doc = F.normalize((h.float() * mf).sum(dim=1) / mf.sum(dim=1).clamp(min=1.0), p=2, dim=-1)
                else:
                    last_pos = mask_t.sum(dim=1) - 1
                    doc = F.normalize(h[torch.arange(len(b), device=dev), last_pos].float(), p=2, dim=-1)
            arr = doc.cpu().numpy()
            for k, idx in enumerate(b):
                out[idx] = arr[k]
        return out

    res: dict[str, dict] = {"meta": {"tag": f"b0_{backbone}", "backbone": backbone, "tau_a": tau_a_override}, "B0_doc": {}}
    for bname, fname, cb_rel, cb_npy, is_drg in [
        ("source_icd9_test", "source_icd9_test.jsonl.gz", "benchmarks/mimiciv-2.2/codebooks/icd9-full.json", "icd9-full.npy", False),
        ("mimiciv-icd10_test", "icd10_test.jsonl.gz", None, "mimiciv-icd10.npy", False),
        ("mimiciv-ms-drg_test", "drg_test.jsonl.gz", None, "mimiciv-ms-drg.npy", True),
        ("codiesp_test", "codiesp_test.jsonl.gz", None, "codiesp.npy", False),
    ]:
        rows = _load_jsonl_gz(test_dir / fname)
        cb_path = (data_root / cb_rel) if cb_rel else target_paths(bname.replace("_test", ""), data_root)["codebook"]
        cb = load_codebook(cb_path)
        k2i = {e.key: i for i, e in enumerate(cb)}
        z = l2_normalize(np.load(cb_dir / cb_npy).astype(np.float32))
        embs = _encode_doc([r["text"] for r in rows])
        sc = embs @ z.T
        if is_drg:
            m, _ = evaluate_answer_groups(sc, [tuple(tuple(g) for g in r["label_groups"]) for r in rows], k2i)
        else:
            m, _ = evaluate_multilabel(sc, [tuple(r["labels"]) for r in rows], k2i, threshold=tau_a_override)
        res["B0_doc"][bname] = m
        if bname == "source_icd9_test":
            fkeys = _load_key_subset(data_root / "benchmarks/mimiciv-2.2/codebooks/icd9-frequent.json")
            fidx = np.array([k2i[k] for k in fkeys if k in k2i], dtype=np.int32)
            fk2i = {cb[j].key: i for i, j in enumerate(fidx)}
            fg = [tuple(k for k in r["labels"] if k in fk2i) for r in rows]
            mf, _ = evaluate_multilabel(sc[:, fidx], fg, fk2i, threshold=tau_a_override)
            res["B0_doc"]["source_icd9_test_frequent"] = mf
        elif bname == "mimiciv-icd10_test":
            f10 = _load_key_subset(target_paths("mimiciv-icd10", data_root)["frequent_codebook"])
            ridx = np.array([i for i, e in enumerate(cb) if e.key not in f10], dtype=np.int32)
            rk2i = {cb[j].key: i for i, j in enumerate(ridx)}
            rg = [tuple(k for k in r["labels"] if k in rk2i) for r in rows]
            mr, _ = evaluate_multilabel(sc[:, ridx], rg, rk2i, threshold=tau_a_override)
            res["B0_doc"]["mimiciv-icd10_test_rare"] = mr
        print(f"  [B0 {backbone}] {bname}: {m}", flush=True)

    res["meta"]["elapsed_sec"] = round(time.perf_counter() - t0, 1)
    out_path = out_dir / f"sealed_test_b0_{backbone}.json"
    out_path.write_text(json.dumps(res, indent=2), encoding="utf-8")
    del bb
    torch.cuda.empty_cache()
    return res


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["ocsf", "closed_laat", "b0"], required=True)
    parser.add_argument("--tag", type=str, required=True)
    parser.add_argument("--backbone", type=str, required=True)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--tau-a", type=float, default=0.30)
    parser.add_argument("--include-unnorm", action="store_true")
    parser.add_argument("--test-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--max-eval-tokens", type=int, default=65536)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.mode == "ocsf":
        eval_ocsf_checkpoint(
            args.tag, args.backbone, args.run_dir, args.test_dir, args.cache_dir, args.data_root,
            args.out_dir, dev, include_unnorm=args.include_unnorm, max_eval_tokens=args.max_eval_tokens,
        )
    elif args.mode == "closed_laat":
        eval_closed_laat(args.run_dir, args.test_dir, args.data_root, args.out_dir, dev, max_eval_tokens=args.max_eval_tokens)
    elif args.mode == "b0":
        eval_b0_doc(args.backbone, args.test_dir, args.cache_dir, args.data_root, args.out_dir, dev, tau_a_override=args.tau_a, max_eval_tokens=args.max_eval_tokens)


if __name__ == "__main__":
    main()
