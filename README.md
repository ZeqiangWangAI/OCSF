# OCSF: Ontology-Compiled 2D Clinical State Fields

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-3776AB.svg)](https://www.python.org/)

Official implementation of **"Ontology-Compiled 2D Clinical State Fields with Bidirectional Diffusion Encoders for Zero-Training Cross-Ontology Medical Coding"**.

**Author:** Zeqiang Wang (Department of Computer Science, University of Surrey; `zeqiang.wang@surrey.ac.uk`).

---

## Overview

Healthcare systems continually revise clinical coding taxonomies, update Diagnosis-Related Group (DRG) reimbursement schedules, and harmonize multi-center registries across languages. Standard neural medical coders bind clinical text understanding to fixed, code-specific classification heads, requiring manual chart relabeling and neural retraining whenever target vocabularies change.

**Ontology-Compiled 2D Clinical State Fields (OCSF)** decouple clinical document comprehension from target ontology indexing:

1. **Online Document Encoding (Zero Code-ID Parameters):** A half-frozen open-weight bidirectional diffusion-pretrained encoder (`perplexity-ai/pplx-embed-context-v1-{0.6B,4B}`) and a vocabulary-free token-channel decoder map an unchunked discharge summary ($L \le 3{,}072$ tokens) into an ontology-invariant two-dimensional clinical state matrix $\hat{U}(x) \in \mathbb{R}^{R \times K}$ ($R=512$ semantic channels).
2. **Offline Ontology Compilation (`< 10 s` on CPU):** Any unseen target ontology $B = \{(c_j, d_j^B)\}_{j=1}^{|B|}$ is compiled from its public text descriptions into a closed-form linear readout operator $W_B$ via **Scale-Invariant Self-Normalized Tikhonov Deconvolution** (solved in $\mathcal{O}(n_r^3)$ time via the exact Woodbury dual), suppressing sibling-code cross-talk while preserving source-calibrated decision thresholds $\tau_A$ across a $1{,}600\times$ span of vocabulary sizes.
3. **Compositional Rule Compilation:** Public healthcare rules—including principal-diagnosis background debiasing, CMS MS-DRG v25–v37 complication/surgical modifiers, and procedural equivalence groups—compile algebraically into the linear readout operator without learned target parameters.

---

## Headline Empirical Results

Trained **exclusively on English MIMIC-IV v2.2 ICD-9** (`154,542` notes) and evaluated in a single locked pass across `58,706` held-out test notes and five unseen ontologies (`115` to `185,458` codes) with **zero target training and zero threshold tuning**:

| Ontology | Language | Codes $\|B\|$ | Task | Primary Result | CPU Compile | GPU Readout |
| :--- | :--- | ---: | :--- | :--- | ---: | ---: |
| **MIMIC-IV ICD-9** *(Source)* | English | 11,324 (6,150 freq.) | Multi-label | **63.15% Micro-F1** / **70.98% P@8** *(vs. 62.10% Closed LAAT)* | 0.145 s | 0.18 ms/note |
| **MIMIC-IV ICD-10** *(Unseen)* | English | 26,085 | Multi-label | **29.23% R@\|y\|** / **26.12% F1** (**32.38%** ICD-10-CM) | 0.800 s | 0.13 ms/note |
| **MIMIC-IV MS-DRG** *(Unseen)* | English | 1,581 | Single-label | **7.41%–9.91% Top-1** / **22.64%–24.92% Top-5** | 0.040 s | 0.05 ms/note |
| **CodiEsp** *(Unseen)* | Spanish | 185,458 | Multi-label | **11.28% R@\|y\|** / **14.50% P@8** | 9.681 s | 0.17 ms/note |
| **ElCardioCC** *(Unseen)* | Greek | 115 | Single-label | **73.04% Top-1** / **97.04% Top-5** | 0.028 s | 0.13 ms/note |
| **CHIP-CDN** *(Unseen)* | Chinese | 37,645 | Multi-label | **22.59%–26.01% R@\|y\|** *(zero 1D rank collapse)* | 0.799 s | 0.12 ms/note |

---

## Installation

```bash
git clone https://github.com/ZeqiangWangAI/OCSF.git
cd OCSF

# Recommended: install via uv (or standard pip venv)
uv venv .venv --python 3.12
source .venv/bin/activate
uv pip install -e ".[dev]"

# Verify installation and mathematical invariants (runs on CPU in <2s without clinical data)
pytest -v
```

---

## Minimal Quickstart: Compiling Any Public Codebook on CPU

```python
import numpy as np
from ocsf.state_matrix import ChannelBank, compile_deconvolution_operator

# 1. Fit R semantic channel anchors once on unit-normalized source codebook embeddings Z_A
rng = np.random.default_rng(42)
z_source = rng.normal(size=(11324, 1024)).astype(np.float32)
bank = ChannelBank.fit(z_source, n_channels=512, seed=42)

# 2. Switch to any unseen target codebook Z_B on CPU in milliseconds (zero target labels)
z_target = rng.normal(size=(26085, 1024)).astype(np.float32)
operator = compile_deconvolution_operator(z_target, bank=bank, lam=0.15, self_normalize=True)

# 3. Score predicted 2D clinical states U_hat of shape (batch_size, R, K)
u_hat = rng.normal(size=(8, 512, 1024)).astype(np.float32)
scores = operator.score_batch(u_hat)  # shape: (8, 26085)
```

---

## Repository Structure

```text
├── src/ocsf/
│   ├── state_matrix.py    # Spherical k-means ChannelBank & Woodbury Self-Normalized Deconvolution
│   ├── e2e.py             # Bidirectional diffusion / causal encoders, LoRA, decoder, & Closed LAAT
│   ├── compiler.py        # Principal-state debiasing, CMS MS-DRG rules, & OR-group pooling
│   ├── models.py          # Output-space conditional Rectified Flow & chunked state baselines
│   ├── baselines.py       # Zero-shot bi-encoder & 1D analytic projection baselines
│   ├── encode.py          # Frozen codebook & clinical text embedding utilities
│   ├── data.py            # Patient-disjoint & held-out split loaders (via $OCSF_DATA_ROOT)
│   └── metrics.py         # OOV-preserving R@|y|, F1, paired bootstrap CIs, Cohen's d_z & h
├── scripts/
│   ├── run_encode.py                     # Precompute frozen codebook & document embeddings
│   ├── run_train_e2e_ocsf.py             # Source-only MIMIC-IV ICD-9 training & zero-shot dev eval
│   ├── run_eval_sealed_test.py           # One-shot locked held-out test evaluation
│   ├── run_stats_and_multiseed_audit.py  # Paired effect sizes (d_z, h) & 5-seed partition audit
│   ├── run_eval_reviewer_suite.py        # Latency/VRAM, perturbation, & CMS v25-v37 diagnostics
│   ├── run_train_flow.py                 # Output-space Rectified Flow mechanism ablation
│   └── isambard/                         # Slurm batch scripts for GH200 training & evaluation
└── tests/                 # Unit & integration test suite
```

---

## Reproducing Training and Evaluation

### 1. Data Preparation
Set `OCSF_DATA_ROOT` to your local benchmark directory containing credentialed PhysioNet MIMIC-IV v2.2 notes and public target codebooks (CodiEsp, ElCardioCC, CHIP-CDN/CBLUE):

```bash
export OCSF_DATA_ROOT=/path/to/clinical-benchmarks/data
```

- **MIMIC-IV v2.2 / MIMIC-IV-Note:** Obtain credentialed access via [PhysioNet](https://physionet.org/content/mimiciv/2.2/) and apply the patient-disjoint splits following Edin et al. (SIGIR 2023).
- **CodiEsp (ICD-10-ES):** Available via [Zenodo / CLEF eHealth 2020](https://zenodo.org/records/3837305).
- **ElCardioCC (Greek Cardiology ICD-10):** Available via [BioASQ / CLEF 2025](http://bioasq.org/).
- **CHIP-CDN (Chinese ICD-10-CN):** Available via [CBLUE](https://github.com/CBLUEbenchmark/CBLUE).

### 2. End-to-End Source Training & Zero-Training Cross-Ontology Transfer
```bash
# Precompute frozen codebook embeddings once
PYTHONPATH=src python scripts/run_encode.py \
  --backbone pplx-0.6b --cache-dir ./cache --codebooks-only

# Train OCSF (0.6B or 4B) exclusively on English MIMIC-IV ICD-9
PYTHONPATH=src python scripts/run_train_e2e_ocsf.py \
  --backbone pplx-0.6b \
  --cache-dir ./cache \
  --out-dir ./runs/ocsf-pplx-0.6b \
  --n-channels 512 --lora-r 32 --freeze-ratio 0.5 --lam 0.15 --epochs 6
```

### 3. Locked Evaluation, Effect Sizes, and Multi-Seed Partition Audit
```bash
PYTHONPATH=src python scripts/run_eval_sealed_test.py --help
PYTHONPATH=src python scripts/run_stats_and_multiseed_audit.py --help
```

---

## Citation

```bibtex
@article{wang2026ocsf,
  title   = {Ontology-Compiled 2D Clinical State Fields with Bidirectional Diffusion Encoders for Zero-Training Cross-Ontology Medical Coding},
  author  = {Wang, Zeqiang},
  year    = {2026},
  url     = {https://github.com/ZeqiangWangAI/OCSF}
}
```

---

## License

This project is licensed under the **GNU Affero General Public License v3.0 (AGPL-3.0)** — see [LICENSE](LICENSE) for full terms. Any modification, derivative work, or network/cloud service deployment interacting with users over a computer network must make its complete corresponding source code publicly available under AGPL-3.0.

Pretrained encoder weights loaded dynamically from Hugging Face Hub remain governed by their respective upstream licenses (`perplexity-ai/pplx-embed-context-v1-{0.6B,4B}` under MIT License; `Qwen/Qwen3-Embedding-{0.6B,4B}` under Apache-2.0 License).
