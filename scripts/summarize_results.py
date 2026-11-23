"""Generate clean Markdown and LaTeX result tables from OCSF experiment summaries.

Reads JSON summaries produced by:
  - scripts/run_eval_b0_and_baselines.py  (B0_doc, B0_chunk_max, B1_CAST_1D, B1_1D_State)
  - scripts/run_train_ocsf.py             (Frozen-chunk V2.1 / V3 OCSF)
  - scripts/run_train_e2e_ocsf.py         (OCSF-E2E & Closed-Vocab LAAT)
  - scripts/run_eval_e2e_checkpoint.py    (Zero-retrain compiler & LoRA-scale ablations)

Also tabulates cited literature references on MIMIC-IV ICD-9 (6,150 frequent codes, Edin et al.
SIGIR 2023, Table 5, arXiv:2304.10909) strictly labeled as published reference numbers.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

# Published reference numbers from Edin et al. (SIGIR 2023, arXiv:2304.10909, Table 5)
# on MIMIC-IV ICD-9 (6,150 frequent codes, freq >= 10) — closed-vocabulary supervised models.
EDIN_2023_MIMIC4_ICD9_FREQ = [
    ("CNN (Edin'23)", 52.4, None, None, None),
    ("Bi-GRU (Edin'23)", 55.5, None, None, None),
    ("CAML (Edin'23)", 58.6, None, None, None),
    ("MultiResCNN (Edin'23)", 60.4, None, None, None),
    ("LAAT (Edin'23)", 61.7, None, None, None),
    ("PLM-ICD (Edin'23)", 62.6, 62.7, 70.0, 53.5),
]

BENCH_ORDER = (
    ("source_icd9_val_frequent", "ICD-9 (Edin 6,150)", "multilabel"),
    ("source_icd9_val", "ICD-9 (Full 11,324)", "multilabel"),
    ("mimiciv-icd10", "MIMIC ICD-10 (26k)", "multilabel"),
    ("mimiciv-icd10-frequent", "ICD-10 Freq (7.9k)", "multilabel"),
    ("mimiciv-icd10-rare", "ICD-10 Rare (18.1k)", "multilabel"),
    ("mimiciv-ms-drg", "MS-DRG (1.5k)", "group"),
    ("codiesp", "CodiEsp ES (185k)", "multilabel"),
    ("elcardiocc-2026", "ElCardioCC EL (115)", "group"),
    ("chip-cdn", "CHIP-CDN ZH (37.6k)", "multilabel"),
)


def _fmt_pct(val: float | None) -> str:
    return "—" if val is None else f"{val * 100:.2f}"


def _fmt_cell(m: dict[str, Any] | None, kind: str) -> str:
    if not m:
        return "—"
    if kind == "group":
        return f"{m['acc@1']*100:.2f} / {m['acc@5']*100:.2f}"
    r_y = m.get("r@|y|")
    f1 = m.get("micro_f1@tau_A")
    return f"{_fmt_pct(r_y)} / {_fmt_pct(f1)}"


def build_markdown_report(entries: list[tuple[str, dict[str, Any]]]) -> str:
    lines: list[str] = ["# OCSF Experiment Summary\n"]

    # Table 1: In-domain MIMIC-IV ICD-9 (Edin 6,150 frequent + Full 11,324)
    lines.append("## 1. In-Domain MIMIC-IV ICD-9 (vs Published Closed-Vocab Baselines)\n")
    lines.append("| Model | Zero-Train Transfer? | Freq-6150 Micro-F1 | Freq-6150 R@|y| | Freq-6150 P@8 | Full-11324 Micro-F1 | Full-11324 P@15 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for name, f1, ry, p8, p15 in EDIN_2023_MIMIC4_ICD9_FREQ:
        s_ry = "—" if ry is None else f"{ry:.1f}"
        s_p8 = "—" if p8 is None else f"{p8:.1f}"
        s_p15 = "—" if p15 is None else f"{p15:.1f}"
        lines.append(f"| {name} | No (closed vocab) | {f1:.1f} | {s_ry} | {s_p8} | — | {s_p15} |")
    for label, res in entries:
        fq = res.get("source_icd9_val_frequent", {})
        fl = res.get("source_icd9_val", {})
        if not fq and not fl:
            continue
        zt = "No (closed vocab)" if "closed" in label.lower() else "Yes (matrix compile)"
        lines.append(
            f"| {label} | {zt} | {_fmt_pct(fq.get('micro_f1@tau_A'))} | {_fmt_pct(fq.get('r@|y|'))} | "
            f"{_fmt_pct(fq.get('p@8'))} | {_fmt_pct(fl.get('micro_f1@tau_A'))} | {_fmt_pct(fl.get('p@15'))} |"
        )

    # Table 2: Zero-training multi-ontology & multilingual transfer across all 5 frozen targets
    lines.append("\n## 2. Zero-Training Cross-Ontology & Cross-Lingual Transfer (R@|y| / F1@tauA; acc@1 / acc@5 for DRG & ElCardioCC)\n")
    tgt_cols = [b for b in BENCH_ORDER if not b[0].startswith("source_")]
    header = "| Model | " + " | ".join(title for _, title, _ in tgt_cols) + " |"
    sep = "|---|" + "|".join("---:" for _ in tgt_cols) + "|"
    lines.extend([header, sep])
    for label, res in entries:
        if all(key not in res for key, _, _ in tgt_cols):
            continue
        cells = [_fmt_cell(res.get(key), kind) for key, _, kind in tgt_cols]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", action="append", required=True, help="Label=path/to/summary.json[:variant]")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    entries: list[tuple[str, dict[str, Any]]] = []
    for item in args.spec:
        label, rest = item.split("=", 1)
        if ":" in rest:
            path_s, variant = rest.rsplit(":", 1)
        else:
            path_s, variant = rest, None
        raw = json.loads(Path(path_s).read_text(encoding="utf-8"))
        if "results" in raw and isinstance(raw["results"], dict):
            raw = raw["results"]
        if variant is not None:
            res = raw[variant]
        else:
            res = raw
        entries.append((label, res))

    md = build_markdown_report(entries)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
