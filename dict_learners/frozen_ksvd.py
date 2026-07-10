"""
Frozen K-SVD: Dictionary Learning with Fixed (Frozen) Atoms
============================================================

A variant of the K-SVD algorithm where a subset of dictionary atoms are
kept fixed ("frozen") during the dictionary update stage.  Only the
remaining "learnable" atoms are updated via rank-1 SVD approximation.

Base algorithm reference:
    M. Aharon, M. Elad, and A. Bruckstein,
    "K-SVD: An Algorithm for Designing Overcomplete Dictionaries
     for Sparse Representation,"
    IEEE Trans. Signal Process., vol. 54, no. 11, pp. 4311-4322, 2006.

The dictionary D is partitioned as:

    D = [D_frozen | D_learnable]

During the dictionary update stage, only columns in D_learnable are
updated.  Frozen columns participate in sparse coding (OMP) but are
never modified.

Public API follows the sklearn estimator convention used by the
``ksvd.ApproximateKSVD`` package so that this class is a drop-in
replacement inside existing pipelines.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class FrozenKSVD:
    """Frozen K-SVD dictionary learning with sklearn-compatible API.

    Parameters
    ----------
    n_components : int
        Total number of dictionary atoms (frozen + learnable).
    n_frozen : int
        Number of leading atoms in the dictionary to keep fixed.
        Must be strictly less than ``n_components``.
    max_iter : int
        Maximum number of outer K-SVD iterations.
    tol : float
        Early-stop tolerance on relative change in reconstruction error.
    transform_n_nonzero_coefs : int
        Maximum non-zero coefficients per signal in OMP sparse coding.
    random_state : int or None
        Seed for reproducible initialisation.

    Attributes
    ----------
    components_ : ndarray of shape (n_components, n_features)
        The learned dictionary.  Row ``k`` is atom ``k``.
        Rows ``0 .. n_frozen-1`` are identical to the frozen atoms
        supplied (or generated) at fit time.
    n_iter_ : int
        Number of iterations actually executed.
    error_ : list[float]
        Reconstruction error at each iteration.
    """

    def __init__(
        self,
        n_components: int = 128,
        n_frozen: int = 0,
        max_iter: int = 10,
        tol: float = 1e-6,
        transform_n_nonzero_coefs: int = 10,
        random_state: Optional[int] = None,
    ) -> None:
        if n_frozen < 0:
            raise ValueError("n_frozen must be >= 0.")
        if n_frozen >= n_components:
            raise ValueError("n_frozen must be strictly less than n_components.")
        if transform_n_nonzero_coefs < 1:
            raise ValueError("transform_n_nonzero_coefs must be >= 1.")

        self.n_components = n_components
        self.n_frozen = n_frozen
        self.max_iter = max_iter
        self.tol = tol
        self.transform_n_nonzero_coefs = transform_n_nonzero_coefs
        self.random_state = random_state

        self.components_: Optional[NDArray[np.floating]] = None
        self.n_iter_: int = 0
        self.error_: list[float] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        X: NDArray[np.floating],
        frozen_atoms: Optional[NDArray[np.floating]] = None,
    ) -> "FrozenKSVD":
        """Learn the dictionary from training data.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training signals.  Each row is one signal.
        frozen_atoms : ndarray of shape (n_frozen, n_features), optional
            Pre-defined atoms to freeze.  If ``None`` and ``n_frozen > 0``,
            an overcomplete DCT basis is used.

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=np.float64)
        n_samples, n_features = X.shape

        # Internally the algorithm works with (n_features, n_samples) layout
        # (columns are signals).
        signals = X.T

        rng = np.random.default_rng(self.random_state)

        # ---- Initialise dictionary (n_features, n_components) ----
        dictionary = np.empty((n_features, self.n_components))

        # Frozen partition
        if self.n_frozen > 0:
            if frozen_atoms is not None:
                frozen_atoms = np.asarray(frozen_atoms, dtype=np.float64)
                if frozen_atoms.shape != (self.n_frozen, n_features):
                    raise ValueError(
                        f"frozen_atoms shape {frozen_atoms.shape} doesn't "
                        f"match expected ({self.n_frozen}, {n_features})."
                    )
                dictionary[:, : self.n_frozen] = frozen_atoms.T
            else:
                dictionary[:, : self.n_frozen] = _build_dct_atoms(
                    n_features, self.n_frozen
                )

        # Learnable partition: random training signals, normalised
        n_learnable = self.n_components - self.n_frozen
        pool_size = min(n_samples, n_learnable)
        random_indices = rng.choice(n_samples, size=pool_size, replace=False)
        dictionary[:, self.n_frozen : self.n_frozen + pool_size] = signals[
            :, random_indices
        ]
        if pool_size < n_learnable:
            dictionary[:, self.n_frozen + pool_size :] = rng.standard_normal(
                (n_features, n_learnable - pool_size)
            )

        dictionary = _normalise_columns(dictionary)

        # ---- Main K-SVD loop ----
        self.error_ = []
        prev_error = np.inf

        for iteration in range(1, self.max_iter + 1):
            # Stage 1: Sparse coding (OMP over full dictionary)
            codes = _sparse_coding(
                dictionary, signals, self.transform_n_nonzero_coefs
            )

            # Stage 2: Dictionary update (skip frozen atoms)
            dictionary, codes = _update_dictionary(
                dictionary, signals, codes, self.n_frozen
            )

            # Re-normalise learnable atoms only
            for k in range(self.n_frozen, self.n_components):
                col_norm = np.linalg.norm(dictionary[:, k])
                if col_norm > 1e-12:
                    dictionary[:, k] /= col_norm

            # Track reconstruction error
            recon_error = float(
                np.linalg.norm(signals - dictionary @ codes, "fro")
            )
            self.error_.append(recon_error)
            logger.info(
                "Iteration %d / %d — error: %.6e",
                iteration,
                self.max_iter,
                recon_error,
            )

            # Early stopping
            rel_change = abs(prev_error - recon_error) / max(prev_error, 1e-12)
            if rel_change < self.tol:
                logger.info(
                    "Converged at iteration %d (rel_change=%.2e).",
                    iteration,
                    rel_change,
                )
                break
            prev_error = recon_error

        self.n_iter_ = len(self.error_)

        # Store in row-major (n_components, n_features) to match
        # the sklearn / ApproximateKSVD convention.
        self.components_ = dictionary.T

        return self

    def transform(
        self,
        X: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """Encode new signals against the learned dictionary.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Signals to encode.

        Returns
        -------
        codes : ndarray of shape (n_samples, n_components)
            Sparse coefficient matrix.
        """
        if self.components_ is None:
            raise RuntimeError("fit() must be called before transform().")

        X = np.asarray(X, dtype=np.float64)
        signals = X.T  # (n_features, n_samples)
        dictionary = self.components_.T  # (n_features, n_components)

        codes = _sparse_coding(
            dictionary, signals, self.transform_n_nonzero_coefs
        )
        return codes.T  # (n_samples, n_components)


# ======================================================================
# Internal helpers — algorithm logic unchanged
# ======================================================================


def _omp(
    dictionary: NDArray[np.floating],
    signal: NDArray[np.floating],
    sparsity: int,
) -> NDArray[np.floating]:
    """Encode a single signal using Orthogonal Matching Pursuit.

    Solves:  min_x ||signal - dictionary @ x||_2  s.t. ||x||_0 <= sparsity
    """
    n_atoms = dictionary.shape[1]
    residual = signal.copy()
    support: list[int] = []
    x = np.zeros(n_atoms)

    for _ in range(sparsity):
        correlations = dictionary.T @ residual
        best_atom = int(np.argmax(np.abs(correlations)))

        if best_atom in support:
            break
        support.append(best_atom)

        D_support = dictionary[:, support]
        coeffs, *_ = np.linalg.lstsq(D_support, signal, rcond=None)
        residual = signal - D_support @ coeffs

        if np.linalg.norm(residual) < 1e-12:
            break

    if support:
        D_support = dictionary[:, support]
        coeffs, *_ = np.linalg.lstsq(D_support, signal, rcond=None)
        x[support] = coeffs

    return x


def _sparse_coding(
    dictionary: NDArray[np.floating],
    signals: NDArray[np.floating],
    sparsity: int,
) -> NDArray[np.floating]:
    """Sparse-code every column of ``signals`` against ``dictionary``.

    Parameters
    ----------
    dictionary : (n_features, n_atoms)
    signals : (n_features, n_samples)
    sparsity : max non-zeros per signal

    Returns
    -------
    codes : (n_atoms, n_samples)
    """
    n_atoms = dictionary.shape[1]
    n_signals = signals.shape[1]
    codes = np.zeros((n_atoms, n_signals))

    for i in range(n_signals):
        codes[:, i] = _omp(dictionary, signals[:, i], sparsity)

    return codes


def _update_dictionary(
    dictionary: NDArray[np.floating],
    signals: NDArray[np.floating],
    codes: NDArray[np.floating],
    n_frozen: int,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Update only the learnable (non-frozen) atoms using rank-1 SVD.

    Atoms with indices ``0 .. n_frozen-1`` are never modified.
    """
    n_atoms = dictionary.shape[1]

    for k in range(n_frozen, n_atoms):
        omega_k = np.nonzero(codes[k, :])[0]

        if len(omega_k) == 0:
            # Replace dead atom with the worst-reconstructed signal
            errors = signals - dictionary @ codes
            worst_idx = int(np.argmax(np.linalg.norm(errors, axis=0)))
            dictionary[:, k] = errors[:, worst_idx]
            norm = np.linalg.norm(dictionary[:, k])
            if norm > 1e-12:
                dictionary[:, k] /= norm
            continue

        codes[k, :] = 0.0
        error_matrix = signals - dictionary @ codes

        error_restricted = error_matrix[:, omega_k]

        U, S, Vt = np.linalg.svd(error_restricted, full_matrices=False)

        dictionary[:, k] = U[:, 0]
        codes[k, omega_k] = S[0] * Vt[0, :]

    return dictionary, codes


def _normalise_columns(
    matrix: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Normalise each column to unit L2 norm."""
    norms = np.linalg.norm(matrix, axis=0, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return matrix / norms


def _build_dct_atoms(
    signal_dim: int,
    n_atoms: int,
) -> NDArray[np.floating]:
    """Build an overcomplete DCT-II basis truncated to ``n_atoms`` columns."""
    atoms = np.zeros((signal_dim, n_atoms))
    for k in range(n_atoms):
        for n in range(signal_dim):
            atoms[n, k] = np.cos(np.pi * (2 * n + 1) * k / (2 * signal_dim))
    return _normalise_columns(atoms)
