"""Single-pass long-text late-chunking encoders for controlled backbone pairs.

Controlled backbone pairs (same Qwen3 base, same native dim d, same 8,192-token budget):
  - Diffusion bidirectional: perplexity-ai/pplx-embed-context-v1-{0.6B,4B} (continuous pre-quant tanh)
  - Causal autoregressive:   Qwen/Qwen3-Embedding-{0.6B,4B} (exact same chunk boundaries)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


@dataclass(frozen=True)
class BackboneSpec:
    name: str
    hf_id: str
    family: str  # "pplx_diffusion" | "qwen_causal"
    dim: int
    trust_remote_code: bool


BACKBONE_SPECS: dict[str, BackboneSpec] = {
    "pplx-0.6b": BackboneSpec(
        name="pplx-0.6b",
        hf_id="perplexity-ai/pplx-embed-context-v1-0.6B",
        family="pplx_diffusion",
        dim=1024,
        trust_remote_code=True,
    ),
    "qwen-0.6b": BackboneSpec(
        name="qwen-0.6b",
        hf_id="Qwen/Qwen3-Embedding-0.6B",
        family="qwen_causal",
        dim=1024,
        trust_remote_code=False,
    ),
    "pplx-4b": BackboneSpec(
        name="pplx-4b",
        hf_id="perplexity-ai/pplx-embed-context-v1-4B",
        family="pplx_diffusion",
        dim=2560,
        trust_remote_code=True,
    ),
    "qwen-4b": BackboneSpec(
        name="qwen-4b",
        hf_id="Qwen/Qwen3-Embedding-4B",
        family="qwen_causal",
        dim=2560,
        trust_remote_code=False,
    ),
}


def split_text_into_chunks(
    text: str,
    tokenizer,
    chunk_tokens: int = 256,
    max_tokens: int = 8192,
    max_chunks: int = 32,
) -> list[str]:
    """Language-agnostic deterministic chunker preserving line/paragraph boundaries within token budget."""
    text = text.strip()
    if not text:
        return [" "]
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if len(token_ids) <= chunk_tokens:
        return [text]

    budget = min(len(token_ids), max_tokens - max_chunks)
    token_ids = token_ids[:budget]
    chunks: list[str] = []
    for start in range(0, len(token_ids), chunk_tokens):
        if len(chunks) >= max_chunks:
            break
        piece = tokenizer.decode(token_ids[start : start + chunk_tokens], skip_special_tokens=True).strip()
        if piece:
            chunks.append(piece)
    return chunks if chunks else [" "]


def _mean_pool(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = mask.float().unsqueeze(-1)
    return (hidden * m).sum(dim=1) / m.sum(dim=1).clamp(min=1e-9)


class ClinicalEncoder:
    """Unified single-pass encoder exposing continuous float32 L2-normalized chunk & codebook vectors."""

    def __init__(self, spec_name: str, device: str = "cuda", dtype: torch.dtype = torch.bfloat16) -> None:
        if spec_name not in BACKBONE_SPECS:
            raise ValueError(f"Unknown backbone {spec_name!r}; choose from {sorted(BACKBONE_SPECS)}")
        self.spec = BACKBONE_SPECS[spec_name]
        self.device = torch.device(device)
        padding_side = "left" if self.spec.family == "qwen_causal" else "right"
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.spec.hf_id,
            trust_remote_code=self.spec.trust_remote_code,
            padding_side=padding_side,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModel.from_pretrained(
            self.spec.hf_id,
            trust_remote_code=self.spec.trust_remote_code,
            dtype=dtype,
            attn_implementation="sdpa",
        ).to(self.device)
        if hasattr(self.model.config, "use_cache"):
            self.model.config.use_cache = False
        self.model.eval()

    @torch.inference_mode()
    def encode_codebook_texts(self, texts: list[str], batch_size: int = 256, max_length: int = 256) -> np.ndarray:
        """Encode short ontology/code descriptions into continuous float32 unit vectors (N, d)."""
        n = len(texts)
        out = np.empty((n, self.spec.dim), dtype=np.float32)
        order = np.argsort([len(t) for t in texts], kind="stable")
        for start in range(0, n, batch_size):
            batch_idx = order[start : start + batch_size]
            batch = [texts[int(i)] for i in batch_idx]
            enc = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(self.device)
            hidden = self.model(**enc, use_cache=False).last_hidden_state
            if self.spec.family == "pplx_diffusion":
                pooled = torch.tanh(_mean_pool(hidden, enc["attention_mask"]).float())
            else:
                # Qwen3-Embedding official left-padded last-token pooling
                pooled = hidden[:, -1, :].float()
            out[batch_idx] = F.normalize(pooled, p=2, dim=-1).cpu().numpy()
        return out

    @torch.inference_mode()
    def encode_documents_late_chunks(
        self,
        texts: list[str],
        batch_size: int = 8,
        chunk_tokens: int = 256,
        max_tokens: int = 8192,
        max_chunks: int = 32,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Single-pass unchunked 8,192-token encoding with length-sorted batching and late chunking.

        Returns:
            chunks_arr: (N, max_chunks, d) float16 L2-normalized contextualized chunk embeddings.
            lengths:    (N,) int16 valid chunk count per note.
            doc_mean:   (N, d) float32 L2-normalized full-document embedding (for B0/B1/B2).
        """
        n = len(texts)
        chunks_arr = np.zeros((n, max_chunks, self.spec.dim), dtype=np.float16)
        lengths = np.zeros(n, dtype=np.int16)
        doc_mean = np.zeros((n, self.spec.dim), dtype=np.float32)
        # Sort by character length so batches pack similar-length notes (3x-4x less padding waste).
        order = np.argsort([len(t) for t in texts], kind="stable")

        if self.spec.family == "pplx_diffusion":
            sep_id = (
                self.tokenizer.sep_token_id
                if self.tokenizer.sep_token_id is not None
                else self.tokenizer.eos_token_id
            )
            pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else sep_id
            token_budget = max_tokens - max_chunks
            for start in range(0, n, batch_size):
                batch_orig = order[start : start + batch_size]
                batch_texts = [texts[int(i)].strip() or " " for i in batch_orig]
                # Single fast batched Rust tokenization + direct token-ID chunk separation (zero decode/re-encode drift).
                raw_enc = self.tokenizer(
                    batch_texts,
                    add_special_tokens=False,
                    truncation=True,
                    max_length=token_budget,
                )["input_ids"]

                seqs: list[list[int]] = []
                batch_spans: list[list[tuple[int, int]]] = []
                for ids in raw_enc:
                    if not ids:
                        ids = [pad_id]
                    seq: list[int] = []
                    spans: list[tuple[int, int]] = []
                    for c_idx in range(0, len(ids), chunk_tokens):
                        if len(spans) >= max_chunks:
                            break
                        if seq:
                            seq.append(sep_id)
                        s0 = len(seq)
                        piece = ids[c_idx : c_idx + chunk_tokens]
                        seq.extend(piece)
                        spans.append((s0, len(seq)))
                    seqs.append(seq)
                    batch_spans.append(spans)

                max_len = max(len(s) for s in seqs)
                bsz = len(seqs)
                input_ids = torch.full((bsz, max_len), pad_id, dtype=torch.long)
                attn_mask = torch.zeros((bsz, max_len), dtype=torch.long)
                for b_idx, seq in enumerate(seqs):
                    L = len(seq)
                    input_ids[b_idx, :L] = torch.tensor(seq, dtype=torch.long)
                    attn_mask[b_idx, :L] = 1

                input_ids = input_ids.to(self.device)
                attn_mask = attn_mask.to(self.device)
                hidden = self.model(input_ids=input_ids, attention_mask=attn_mask, use_cache=False).last_hidden_state

                for b_idx, row_i_np in enumerate(batch_orig):
                    row_i = int(row_i_np)
                    spans = batch_spans[b_idx]
                    vecs = []
                    token_sum = torch.zeros(self.spec.dim, device=self.device, dtype=torch.float32)
                    token_cnt = 0
                    for s0, s1 in spans:
                        span_h = hidden[b_idx, s0:s1].float()
                        token_sum += span_h.sum(dim=0)
                        token_cnt += s1 - s0
                        v = torch.tanh(span_h.mean(dim=0))
                        vecs.append(F.normalize(v, p=2, dim=0))
                    stacked = torch.stack(vecs, dim=0)  # (M, d)
                    m = stacked.shape[0]
                    chunks_arr[row_i, :m] = stacked.cpu().numpy().astype(np.float16)
                    lengths[row_i] = m
                    full_v = torch.tanh(token_sum / max(token_cnt, 1))
                    doc_mean[row_i] = F.normalize(full_v, p=2, dim=0).cpu().numpy()
        else:
            # Causal Qwen3-Embedding single-pass causal late-chunking at exact chunk boundaries
            for start in range(0, n, batch_size):
                batch_orig = order[start : start + batch_size]
                batch_texts = [texts[int(i)] for i in batch_orig]
                enc = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=max_tokens,
                    return_tensors="pt",
                ).to(self.device)
                hidden = self.model(**enc, use_cache=False).last_hidden_state
                attn_mask = enc["attention_mask"]
                total_len = hidden.shape[1]

                for b_idx, row_i_np in enumerate(batch_orig):
                    row_i = int(row_i_np)
                    valid_len = int(attn_mask[b_idx].sum().item())
                    pad_offset = total_len - valid_len  # left padding
                    ends = list(range(chunk_tokens, valid_len, chunk_tokens))
                    if not ends or ends[-1] != valid_len:
                        ends.append(valid_len)
                    ends = ends[:max_chunks]

                    vecs = []
                    for e in ends:
                        v = hidden[b_idx, pad_offset + e - 1].float()
                        vecs.append(F.normalize(v, p=2, dim=0))
                    stacked = torch.stack(vecs, dim=0)
                    m = stacked.shape[0]
                    chunks_arr[row_i, :m] = stacked.cpu().numpy().astype(np.float16)
                    lengths[row_i] = m
                    doc_mean[row_i] = stacked[-1].cpu().numpy()

        return chunks_arr, lengths, doc_mean


def save_chunk_cache(path: Path, chunks: np.ndarray, lengths: np.ndarray, doc_mean: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, chunks=chunks, lengths=lengths, doc_mean=doc_mean)
