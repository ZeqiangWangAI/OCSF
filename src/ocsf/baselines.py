"""Standalone Baselines (B0–B3) for controlled, apple-to-apple comparison against OCSF.

- B0: Zero-shot dense retrieval (dot product of doc embedding x_vec with target codebook Z_B).
- B1/B2: 1D Flat Ridge-Projected Bi-Encoder (CAST-style, Heyden et al. 2026) — exact R=1 special
         case of OCSF! Trains a single 1D state vector u(x) in R^K or linear classifier W_A in
         R^{|A| x d}, then projects zero-shot into target Z_B via closed-form ridge:
           P = W_A^T Z_A (Z_A^T Z_A + lam I_K)^{-1}   ->   W_B = Z_B P^T
         giving exact apple-to-apple proof of 2D State Matrix (R=128) vs 1D Vector (R=1).
- B3: Naive 2D Matrix Dot-Product (R=128 without Tikhonov Deconvolution) — directly tests
      whether Tikhonov deconvolution eliminates sibling Gram interference.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .state_matrix import l2_normalize


@dataclass(frozen=True)
class CastProjector:
    """Closed-form CAST zero-shot ridge projection P in R^{d_doc x K} (Heyden et al., 2026)."""

    proj: np.ndarray  # (d_doc, K) float32

    @classmethod
    def fit_from_classifier_weights(
        cls,
        w_source: np.ndarray,  # (|A|, d_doc)
        z_source: np.ndarray,  # (|A|, K)
        lam: float = 0.15,
        active_mask: np.ndarray | None = None,
    ) -> "CastProjector":
        w = w_source.astype(np.float32, copy=False)
        z = l2_normalize(z_source.astype(np.float32, copy=False))
        if active_mask is not None and np.any(active_mask):
            w = w[active_mask]
            z = z[active_mask]
        n = float(max(len(z), 1))
        # Trace-1 normalized ridge so lam is scale-invariant across source vocabulary sizes
        cov = (z.T @ z) / n + lam * np.eye(z.shape[1], dtype=np.float32)
        rhs = (z.T @ w) / n  # (K, d_doc)
        proj = np.linalg.solve(cov, rhs).T.astype(np.float32)  # (d_doc, K)
        return cls(proj=proj)

    def score_target(self, doc_embs: np.ndarray, z_target: np.ndarray) -> np.ndarray:
        """Zero-training scoring on any target codebook B: (doc_embs @ P) @ Z_B^T."""
        z_b = l2_normalize(z_target.astype(np.float32, copy=False))
        latents = doc_embs.astype(np.float32, copy=False) @ self.proj  # (N, K)
        return (latents @ z_b.T).astype(np.float32)


def score_b0_zeroshot_retrieval(doc_embs: np.ndarray, z_target: np.ndarray) -> np.ndarray:
    """B0: Off-the-shelf zero-shot cosine retrieval (doc_embs @ Z_B^T)."""
    d = l2_normalize(doc_embs.astype(np.float32, copy=False))
    z = l2_normalize(z_target.astype(np.float32, copy=False))
    return (d @ z.T).astype(np.float32)
