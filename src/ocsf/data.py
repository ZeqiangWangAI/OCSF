"""Read-only non-test dataset loaders with strict held-out test and patient-disjoint guards.

Reads strictly from untokenized source-text and target prepared JSONL/codebook files under
DATA_ROOT (configured via the OCSF_DATA_ROOT environment variable or ./data by default).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

DEFAULT_DATA_ROOT = Path(os.environ.get("OCSF_DATA_ROOT", "./data"))


def assert_non_test_path(path: Path) -> None:
    """Fail fast if any path points into a sealed or clinical test split."""
    lowered_parts = [p.lower() for p in path.parts]
    for part in lowered_parts:
        if "sealed" in part or part == "test" or part.startswith("test.") or part.endswith("_test.jsonl") or part.endswith("-test.jsonl"):
            raise RuntimeError(f"Blocked access to sealed/test path: {path}")


@dataclass(frozen=True)
class CodebookEntry:
    key: str
    system: str
    code: str
    description: str
    ancestors: tuple[str, ...] = ()

    @property
    def prompt_text(self) -> str:
        # Pure ontology-neutral clinical language (no dataset-ID prefix that leaks source vs target).
        if self.ancestors:
            anc = " > ".join(self.ancestors)
            return f"{anc}: {self.description}"
        return self.description


@dataclass(frozen=True)
class ClinicalExample:
    doc_id: str
    subject_id: str | None
    text: str
    labels: tuple[str, ...]
    label_groups: tuple[tuple[str, ...], ...] = ()


def load_codebook(path: Path) -> list[CodebookEntry]:
    assert_non_test_path(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [
        CodebookEntry(
            key=item["key"],
            system=item["system"],
            code=item["code"],
            description=item["description"],
            ancestors=tuple(item.get("ancestors", ())),
        )
        for item in raw
    ]


def iter_source_icd9(split: str, data_root: Path = DEFAULT_DATA_ROOT, limit: int | None = None) -> Iterator[ClinicalExample]:
    """Stream English MIMIC-IV 2.2 ICD-9 source examples ('train' or 'validation')."""
    if split not in {"train", "validation"}:
        raise ValueError(f"Source ICD-9 only allows 'train' or 'validation', got {split!r}")
    path = data_root / "source-text" / f"{split}.jsonl"
    assert_non_test_path(path)
    with path.open("r", encoding="utf-8") as fh:
        for idx, line in enumerate(fh):
            if limit is not None and idx >= limit:
                break
            row = json.loads(line)
            labels = tuple(
                [f"icd9_diag:{c}" for c in row["icd9_diag"]]
                + [f"icd9_proc:{c}" for c in row["icd9_proc"]]
            )
            yield ClinicalExample(
                doc_id=str(row["_id"]),
                subject_id=str(row["subject_id"]),
                text=row["text"],
                labels=labels,
            )


def load_source_subject_ids(data_root: Path = DEFAULT_DATA_ROOT) -> set[str]:
    """Collect all 82,284 source subject_ids from ICD-9 train+validation to guarantee patient disjointness."""
    subjects: set[str] = set()
    for split in ("train", "validation"):
        path = data_root / "source-text" / f"{split}.jsonl"
        assert_non_test_path(path)
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                row = json.loads(line)
                subjects.add(str(row["subject_id"]))
    return subjects


def target_paths(target_name: str, data_root: Path = DEFAULT_DATA_ROOT) -> dict[str, Path]:
    """Return verified non-test read-only paths for each frozen transfer target."""
    if target_name == "mimiciv-icd10":
        return {
            "eval": data_root / "transfer/mimiciv-icd10-drg/prepared-available/icd10/dev.jsonl",
            "codebook": data_root / "benchmarks/mimiciv-2.2/codebooks/icd10-full.json",
            "frequent_codebook": data_root / "benchmarks/mimiciv-2.2/codebooks/icd10-frequent.json",
        }
    if target_name == "mimiciv-ms-drg":
        return {
            "eval": data_root / "transfer/mimiciv-icd10-drg/drg-public-aliases/dev.jsonl",
            "codebook": data_root / "transfer/mimiciv-icd10-drg/drg-public-aliases/codebook.json",
            "cms_dir": data_root / "transfer/drg/cms-codebooks",
        }
    if target_name == "codiesp":
        return {
            "eval": data_root / "transfer/codiesp/prepared-hierarchical/dev.jsonl",
            "codebook": data_root / "transfer/codiesp/prepared-hierarchical/codebook.json",
            "groups": data_root / "transfer/codiesp/prepared-hierarchical/groups.json",
        }
    if target_name == "elcardiocc-2026":
        return {
            "eval": data_root / "transfer/elcardiocc-2026/prepared/train.jsonl",
            "codebook": data_root / "transfer/elcardiocc-2026/prepared/codebook.json",
        }
    if target_name == "chip-cdn":
        return {
            "eval": data_root / "transfer/chip-cdn/prepared/dev.jsonl",
            "codebook": data_root / "transfer/chip-cdn/prepared/codebook.json",
        }
    raise ValueError(f"Unknown target benchmark: {target_name!r}")


def iter_target_eval(
    target_name: str,
    data_root: Path = DEFAULT_DATA_ROOT,
    source_subjects: set[str] | None = None,
    limit: int | None = None,
) -> Iterator[ClinicalExample]:
    """Stream frozen non-test evaluation examples for a target benchmark.

    Enforces patient-disjointness for MIMIC-IV ICD-10 and MS-DRG automatically.
    """
    paths = target_paths(target_name, data_root)
    eval_path = paths["eval"]
    assert_non_test_path(eval_path)

    needs_disjoint = target_name in {"mimiciv-icd10", "mimiciv-ms-drg"}
    if needs_disjoint and source_subjects is None:
        source_subjects = load_source_subject_ids(data_root)

    emitted = 0
    with eval_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if limit is not None and emitted >= limit:
                break
            row = json.loads(line)
            sid = None if row.get("subject_id") is None else str(row["subject_id"])
            if needs_disjoint and sid is not None and sid in source_subjects:
                continue

            raw_groups = row.get("label_groups") or []
            label_groups = tuple(tuple(g) for g in raw_groups)
            raw_labels = row.get("labels")
            if raw_labels is None:
                # ElCardioCC uses label_groups without a separate flat labels list
                flat = [k for g in label_groups for k in g]
                labels = tuple(dict.fromkeys(flat))
            else:
                labels = tuple(raw_labels)

            emitted += 1
            yield ClinicalExample(
                doc_id=str(row["id"]),
                subject_id=sid,
                text=row["text"],
                labels=labels,
                label_groups=label_groups,
            )
