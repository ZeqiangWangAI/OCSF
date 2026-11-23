"""Offline Target-Codebook Compiler: compiles any public ontology B into pure matrix operators.

Zero target training data, zero target validation labels, zero gradient steps.
Supports:
  - Standard multi-label ICD ontologies (ICD-9, ICD-10, CodiEsp, CHIP-CDN)
  - Narrow-chapter answer-group ontologies (ElCardioCC)
  - Mutually exclusive MS-DRG tables via public CMS structure parsing (base + MCC/CC/None + P/M)
  - Hierarchical OR-group pooling (e.g., CodiEsp 11,381 procedure groups)
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
import numpy as np

from .data import CodebookEntry
from .state_matrix import ChannelBank, DeconvolutionOperator, compile_deconvolution_operator, l2_normalize


_SEVERITY_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("WITHOUT_CC_MCC", re.compile(r"\bWITHOUT\s+CC/MCC\b|\bW/O\s+CC/MCC\b", re.I)),
    ("WITHOUT_MCC", re.compile(r"\bWITHOUT\s+MCC\b|\bW/O\s+MCC\b", re.I)),
    ("WITH_CC_OR_MCC", re.compile(r"\bWITH\s+CC/MCC\b|\bW\s+CC/MCC\b", re.I)),
    ("WITH_MCC", re.compile(r"\bWITH\s+MCC\b|\bW\s+MCC\b", re.I)),
    ("WITH_CC", re.compile(r"\bWITH\s+CC\b|\bW\s+CC\b", re.I)),
)

DRG_PROTOTYPE_PROMPTS: dict[str, str] = {
    "WITH_MCC": "major complication or comorbidity, acute organ failure, septic shock, acute respiratory failure, critical illness",
    "WITH_CC": "complication or comorbidity, secondary chronic or acute clinical condition requiring additional hospital care",
    "WITH_CC_OR_MCC": "complication or comorbidity or major complication or comorbidity during hospitalization",
    "WITHOUT_CC_MCC": "uncomplicated routine hospital stay without complication or comorbidity",
    "WITHOUT_MCC": "hospital stay without major complication or comorbidity",
    "SURGICAL_P": "major operating room surgical procedure, operative intervention under anesthesia",
    "MEDICAL_M": "non-operative medical management without major operating room procedure",
}


def parse_drg_description(description: str) -> tuple[str, str]:
    """Split a public MS-DRG description into (base_condition, severity_tier)."""
    for tier, pat in _SEVERITY_PATTERNS:
        if pat.search(description):
            base = pat.sub("", description).strip(" ,;-")
            return (base if base else description), tier
    return description.strip(), "UNSTRATIFIED"


def load_cms_drg_kinds(cms_dir: Path) -> dict[str, str]:
    """Load public CMS MS-DRG surgical ('P') vs medical ('M') flags keyed by 'code:version'."""
    kinds: dict[str, str] = {}
    if not cms_dir.exists():
        return kinds
    for path in sorted(cms_dir.glob("ms-drg-v*-structure.json")):
        match = re.search(r"v(\d+)", path.name)
        if not match:
            continue
        ver = f"v{match.group(1)}"
        rows = json.loads(path.read_text(encoding="utf-8"))
        for row in rows:
            kinds[f"{row['code']}:{ver}"] = str(row.get("kind", ""))
            kinds.setdefault(str(row["code"]), str(row.get("kind", "")))
    return kinds


@dataclass(frozen=True)
class AlignmentStats:
    """Source-codebook geometry statistics fit ONCE on source A."""

    source_mean: np.ndarray  # (K,) float32
    zca_matrix: np.ndarray | None = None  # (K, K) float32 optional ZCA whitening

    @classmethod
    def fit(cls, z_source: np.ndarray, use_zca: bool = False, zca_shrink: float = 0.1) -> "AlignmentStats":
        z = l2_normalize(z_source.astype(np.float32, copy=False))
        mean = z.mean(axis=0).astype(np.float32)
        if not use_zca:
            return cls(source_mean=mean, zca_matrix=None)
        zc = z - mean[None, :]
        cov = (zc.T @ zc) / max(len(z), 1)
        eigvals, eigvecs = np.linalg.eigh(cov.astype(np.float64))
        eigvals = np.maximum(eigvals, 0.0)
        inv_sqrt = 1.0 / np.sqrt(eigvals + zca_shrink)
        zca = (eigvecs * inv_sqrt[None, :]) @ eigvecs.T
        return cls(source_mean=mean, zca_matrix=zca.astype(np.float32))

    def align_codebook(
        self,
        z_target: np.ndarray,
        centroid_shift_min_codes: int = 1000,
        max_shift_cos: float = 0.90,
    ) -> np.ndarray:
        """Align a target codebook Z_B onto the source clinical frame without destroying same-language or narrow chapters.

        - Broad cross-lingual/cross-domain codebooks (|B| >= 1000 AND cos(mu_A, mu_B) < max_shift_cos):
          removes centroid offset (mu_B - mu_A) so foreign/cross-domain ontologies share source A's mean.
        - Same-language codebooks (cos >= max_shift_cos, e.g. English ICD-10) or narrow single-chapter
          codebooks (|B| < 1000, e.g. ElCardioCC D50-D64): preserves native clinical geometry untouched.
        """
        z = l2_normalize(z_target.astype(np.float32, copy=False))
        if z.shape[0] >= centroid_shift_min_codes:
            target_mean = z.mean(axis=0)
            mu_cos = float(
                np.dot(target_mean, self.source_mean)
                / (np.linalg.norm(target_mean) * np.linalg.norm(self.source_mean) + 1e-8)
            )
            if mu_cos < max_shift_cos:
                z = z - (target_mean[None, :] - self.source_mean[None, :])
        if self.zca_matrix is not None:
            z = (z - self.source_mean[None, :]) @ self.zca_matrix
        return l2_normalize(z)

    def align_anchors_for_target(
        self,
        anchors: np.ndarray,
        z_target: np.ndarray,
        centroid_shift_min_codes: int = 1000,
        max_shift_cos: float = 0.90,
    ) -> np.ndarray:
        """Translate source channel anchors into target B's centroid ONLY on genuine cross-domain/lingual shift.

        Leaves target note chunks H(x) and target codebook vectors Z_B strictly untouched in their
        native multilingual embedding space while aligning channel routing across languages.
        """
        base = l2_normalize(anchors.astype(np.float32, copy=False))
        if z_target.shape[0] < centroid_shift_min_codes:
            return base
        z = l2_normalize(z_target.astype(np.float32, copy=False))
        target_mean = z.mean(axis=0)
        mu_cos = float(
            np.dot(target_mean, self.source_mean)
            / (np.linalg.norm(target_mean) * np.linalg.norm(self.source_mean) + 1e-8)
        )
        if mu_cos >= max_shift_cos:
            return base
        shifted = anchors.astype(np.float32, copy=False) + (target_mean[None, :] - self.source_mean[None, :])
        return l2_normalize(shifted)


def debias_principal_state(
    states: np.ndarray,
    source_mean_state: np.ndarray,
    gamma: float = 0.5,
) -> np.ndarray:
    """Suppress routine multi-label comorbidity background using ONLY source-val mean state.

    Zero target data used: subtracts gamma * E_{x in SourceVal}[U(x)] so single-label
    principal-diagnosis tasks (MS-DRG, ElCardioCC) are not overridden by high-frequency
    ICD-9 comorbidities (e.g., hypertension, hyperlipidemia).
    """
    return (states - float(gamma) * source_mean_state[None, :, :]).astype(np.float32, copy=False)


@dataclass(frozen=True)
class DRGCompositionalOperator:
    """Zero-training algebraic MS-DRG readout factoring base condition, MCC/CC severity, and P/M type."""

    direct_op: DeconvolutionOperator
    prototype_op: DeconvolutionOperator
    prototype_names: tuple[str, ...]
    severity_tiers: tuple[str, ...]  # length |B|
    kinds: tuple[str, ...]  # length |B|, 'P' | 'M' | ''
    severity_weight: float = 0.06
    surgical_weight: float = 0.06

    def combine_scores(self, direct: np.ndarray, proto_scores: np.ndarray) -> np.ndarray:
        """Combine precomputed direct and prototype scores using zero-training severity/surgical rules."""
        name_to_idx = {name: i for i, name in enumerate(self.prototype_names)}
        proto_mean = proto_scores.mean(axis=1, keepdims=True)
        proto_std = np.maximum(proto_scores.std(axis=1, keepdims=True), 1e-4)
        proto_z = (proto_scores - proto_mean) / proto_std

        direct_std = np.maximum(direct.std(axis=1, keepdims=True), 1e-4)
        out = direct.copy()
        for j, (tier, kind) in enumerate(zip(self.severity_tiers, self.kinds)):
            adj = np.zeros(direct.shape[0], dtype=np.float32)
            if tier in name_to_idx:
                adj += self.severity_weight * proto_z[:, name_to_idx[tier]]
            if kind == "P" and "SURGICAL_P" in name_to_idx:
                adj += self.surgical_weight * proto_z[:, name_to_idx["SURGICAL_P"]]
            elif kind == "M" and "MEDICAL_M" in name_to_idx:
                adj += self.surgical_weight * proto_z[:, name_to_idx["MEDICAL_M"]]
            out[:, j] += adj * direct_std[:, 0]
        return out

    def score_batch(self, states: np.ndarray) -> np.ndarray:
        direct = self.direct_op.score_batch(states)  # (N, |B|)
        proto_scores = self.prototype_op.score_batch(states)  # (N, n_proto)
        return self.combine_scores(direct, proto_scores)


def compile_target_operator(
    z_target: np.ndarray,
    bank: ChannelBank,
    alignment: AlignmentStats | None = None,
    lam: float = 0.15,
    self_normalize: bool = True,
    centroid_shift_min_codes: int = 1000,
) -> DeconvolutionOperator:
    """Compile any target codebook Z_B into a native-space Self-Normalized Tikhonov Deconvolution operator.

    When `alignment` is supplied, channel assignment uses centroid-aligned coordinates (so broad
    cross-lingual ontologies route to matching clinical channels) while W_{B,r} is solved strictly
    in Z_B's native embedding space so note chunks H(x) require zero distortion.
    """
    z_native = l2_normalize(z_target.astype(np.float32, copy=False))
    assign_override = None
    if alignment is not None:
        z_for_routing = alignment.align_codebook(z_native, centroid_shift_min_codes=centroid_shift_min_codes)
        assign_override = bank.assign_hard(z_for_routing)
    return compile_deconvolution_operator(
        z_native, bank=bank, lam=lam, self_normalize=self_normalize, assign_override=assign_override
    )


def compile_drg_operator(
    entries: list[CodebookEntry],
    z_target: np.ndarray,
    z_prototypes: dict[str, np.ndarray],
    bank: ChannelBank,
    cms_dir: Path | None = None,
    alignment: AlignmentStats | None = None,
    lam: float = 0.15,
    severity_weight: float = 0.06,
    surgical_weight: float = 0.06,
) -> DRGCompositionalOperator:
    """Compile zero-training MS-DRG compositional operator purely from public CMS metadata."""
    direct_op = compile_target_operator(z_target, bank=bank, alignment=alignment, lam=lam, centroid_shift_min_codes=1000)
    proto_names = tuple(z_prototypes.keys())
    proto_mat = np.stack([z_prototypes[name] for name in proto_names], axis=0).astype(np.float32)
    proto_op = compile_target_operator(proto_mat, bank=bank, alignment=alignment, lam=lam, centroid_shift_min_codes=10**9)

    cms_kinds = load_cms_drg_kinds(cms_dir) if cms_dir is not None else {}
    tiers: list[str] = []
    kinds: list[str] = []
    for entry in entries:
        _, tier = parse_drg_description(entry.description)
        tiers.append(tier)
        # Keys look like 'ms_drg_semantic:057:v25'
        parts = entry.key.split(":")
        kind = ""
        if len(parts) >= 3:
            kind = cms_kinds.get(f"{parts[1]}:{parts[2]}", cms_kinds.get(parts[1], ""))
        elif entry.code:
            kind = cms_kinds.get(entry.code, "")
        kinds.append(kind)

    return DRGCompositionalOperator(
        direct_op=direct_op,
        prototype_op=proto_op,
        prototype_names=proto_names,
        severity_tiers=tuple(tiers),
        kinds=tuple(kinds),
        severity_weight=severity_weight,
        surgical_weight=surgical_weight,
    )


def apply_or_groups(
    scores: np.ndarray,
    code_to_idx: dict[str, int],
    groups_path: Path,
) -> tuple[np.ndarray, dict[str, int]]:
    """Apply exact algebraic OR-group max-pooling (e.g., CodiEsp 11,381 4-char procedure groups).

    Updates existing codebook columns in-place via max(direct_score, max_member_score) without
    duplicating columns; appends only group keys not already present in code_to_idx.
    """
    groups = json.loads(groups_path.read_text(encoding="utf-8"))
    if not groups:
        return scores, code_to_idx

    out = scores.copy()
    new_map = dict(code_to_idx)
    extra_cols: list[np.ndarray] = []

    for grp in groups:
        member_idxs = [code_to_idx[m] for m in grp["members"] if m in code_to_idx]
        if not member_idxs:
            continue
        pooled = scores[:, member_idxs].max(axis=1)
        g_key = grp["key"]
        if g_key in new_map:
            col = new_map[g_key]
            out[:, col] = np.maximum(out[:, col], pooled)
        else:
            new_map[g_key] = out.shape[1] + len(extra_cols)
            extra_cols.append(pooled[:, None])

    if extra_cols:
        out = np.concatenate([out, *extra_cols], axis=1)
    return out, new_map
