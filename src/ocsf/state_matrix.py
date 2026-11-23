"""Ontology-Invariant 2D Clinical State Matrix U in R^{R x K} and Self-Normalized Tikhonov Deconvolution.

Forward-Inverse Duality:
  - Forward (ontology-invariant): u_r(y) = sum_{a in S+_r} z_a in R^K
  - Inverse readout (zero-training, compiled offline from any public codebook Z_B):
      W_{B,r} = diag(g_{B,r})^{-1} Z_{B,r} (Z_{B,r}^T Z_{B,r} + lam I_K)^{-1}
    where g_{B,r,i} = z_i^T (Z_{B,r}^T Z_{B,r} + lam I_K)^{-1} z_i guarantees
    exact unit self-response w_i^T z_i == 1.0 regardless of local sibling density.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from sklearn.cluster import MiniBatchKMeans


def l2_normalize(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(norms, eps)


@dataclass(frozen=True)
class ChannelBank:
    """Fixed ontology-invariant clinical channel anchors C in R^{R x K} fit ONCE on source codebook."""

    anchors: np.ndarray  # (R, K), unit-norm float32

    @property
    def n_channels(self) -> int:
        return int(self.anchors.shape[0])

    @property
    def dim(self) -> int:
        return int(self.anchors.shape[1])

    @classmethod
    def fit(cls, z_source: np.ndarray, n_channels: int, seed: int = 42) -> "ChannelBank":
        z_unit = l2_normalize(z_source.astype(np.float32, copy=False))
        if n_channels == 1:
            center = l2_normalize(z_unit.mean(axis=0, keepdims=True))
            return cls(anchors=center.astype(np.float32))
        km = MiniBatchKMeans(n_clusters=n_channels, random_state=seed, batch_size=2048, n_init=5)
        km.fit(z_unit)
        centers = l2_normalize(km.cluster_centers_.astype(np.float32))
        return cls(anchors=centers)

    def assign_hard(self, z: np.ndarray) -> np.ndarray:
        """Deterministic top-1 channel assignment for any codebook Z in R^{|O| x K}."""
        z_unit = l2_normalize(z.astype(np.float32, copy=False))
        return np.argmax(z_unit @ self.anchors.T, axis=1).astype(np.int32)

    def assign_soft(self, z: np.ndarray, temperature: float = 15.0) -> np.ndarray:
        """Deterministic soft routing weights R_O in R^{|O| x R}."""
        z_unit = l2_normalize(z.astype(np.float32, copy=False))
        logits = temperature * (z_unit @ self.anchors.T)
        logits -= logits.max(axis=1, keepdims=True)
        exp = np.exp(logits)
        return (exp / np.maximum(exp.sum(axis=1, keepdims=True), 1e-8)).astype(np.float32)


@dataclass(frozen=True)
class DeconvolutionOperator:
    """Precompiled zero-training per-channel readout operator for an arbitrary target codebook B."""

    n_codes: int
    n_channels: int
    dim: int
    assign: np.ndarray  # (|B|,) int32
    chan_indices: tuple[np.ndarray, ...]  # length R; global code indices per channel
    chan_weights: tuple[np.ndarray, ...]  # length R; W_{B,r} in R^{m_r x K}
    diag_gains: np.ndarray  # (|B|,) float32 exact leverage g_i in (0, 1]

    def score_batch(self, states: np.ndarray, bias: np.ndarray | None = None) -> np.ndarray:
        """Score a batch of 2D clinical state matrices U in R^{N x R x K} -> S in R^{N x |B|}.

        Uses pure per-channel matrix multiplication; zero neural network evaluation over |B|.
        """
        n_batch = states.shape[0]
        scores = np.empty((n_batch, self.n_codes), dtype=np.float32)
        for r in range(self.n_channels):
            idx = self.chan_indices[r]
            if idx.size == 0:
                continue
            scores[:, idx] = states[:, r, :] @ self.chan_weights[r].T
        if bias is not None:
            scores += bias[None, :]
        return scores


def compile_deconvolution_operator(
    z_codebook: np.ndarray,
    bank: ChannelBank,
    lam: float = 0.15,
    self_normalize: bool = True,
    assign_override: np.ndarray | None = None,
) -> DeconvolutionOperator:
    """Compile scale-invariant Self-Normalized Tikhonov Deconvolution matrices offline from Z_B.

    Uses trace-1 normalized per-channel covariance:
        Sigma_bar_{B,r} = (1 / n_r) * Z_{B,r}^T Z_{B,r}   (Tr(Sigma_bar) == 1.0 for every r and every |B|)
        W_{B,r} = diag(g_{B,r})^{-1} * Z_{B,r} * (Sigma_bar_{B,r} + lam * I_K)^{-1}
    so a single dimensionless shrinkage `lam` works identically from |B|=115 to |B|=185,458
    with bounded condition number <= (1 + lam) / lam and exact unit self-response w_i^T z_i == 1.0.
    """
    z_unit = l2_normalize(z_codebook.astype(np.float32, copy=False))
    n_codes, dim = z_unit.shape
    assign = assign_override.astype(np.int32, copy=False) if assign_override is not None else bank.assign_hard(z_unit)
    eye = np.eye(dim, dtype=np.float32)

    chan_indices: list[np.ndarray] = []
    chan_weights: list[np.ndarray] = []
    diag_gains = np.ones(n_codes, dtype=np.float32)

    for r in range(bank.n_channels):
        idx = np.where(assign == r)[0].astype(np.int32)
        chan_indices.append(idx)
        if idx.size == 0:
            chan_weights.append(np.zeros((0, dim), dtype=np.float32))
            continue
        zr = z_unit[idx]
        n_r = int(idx.size)
        if n_r < dim:
            # Exact Woodbury dual: Z_r (Z_r^T Z_r / n_r + lam I_d)^{-1} = (Z_r Z_r^T / n_r + lam I_{n_r})^{-1} Z_r
            gram = (zr @ zr.T) / float(n_r)
            gram[np.diag_indices(n_r)] += lam
            wr = np.linalg.solve(gram, zr).astype(np.float32)
        else:
            cov = (zr.T @ zr) / float(n_r) + lam * eye
            wr = np.linalg.solve(cov, zr.T).T.astype(np.float32)
        gains = np.sum(wr * zr, axis=1).astype(np.float32)
        diag_gains[idx] = gains
        if self_normalize:
            wr = wr / np.maximum(gains[:, None], 1e-4)
        chan_weights.append(wr)

    return DeconvolutionOperator(
        n_codes=n_codes,
        n_channels=bank.n_channels,
        dim=dim,
        assign=assign,
        chan_indices=tuple(chan_indices),
        chan_weights=tuple(chan_weights),
        diag_gains=diag_gains,
    )


def encode_ground_truth_state(
    label_index_lists: list[np.ndarray],
    z_source: np.ndarray,
    assign: np.ndarray,
    n_channels: int,
) -> np.ndarray:
    """Build exact permutation-free ground-truth 2D Clinical State Matrices U_0(y) in R^{N x R x K}."""
    z_unit = l2_normalize(z_source.astype(np.float32, copy=False))
    dim = z_unit.shape[1]
    out = np.zeros((len(label_index_lists), n_channels, dim), dtype=np.float32)
    for i, idxs in enumerate(label_index_lists):
        if len(idxs) == 0:
            continue
        chans = assign[idxs]
        np.add.at(out[i], chans, z_unit[idxs])
    return out
