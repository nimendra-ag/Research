"""
LC-KSVD: Label Consistent K-SVD
================================
Implementation based on:
  [1] Jiang, Z., Lin, Z., Davis, L.S. (2011).
      "Learning a Discriminative Dictionary for Sparse Coding via Label Consistent K-SVD."
      CVPR 2011, pp. 1697-1704.

  [2] Jiang, Z., Lin, Z., Davis, L.S. (2013).
      "Label Consistent K-SVD: Learning a Discriminative Dictionary for Recognition."
      IEEE TPAMI, 35(11), pp. 2651-2664.

Overview of the method
----------------------
Standard K-SVD learns a dictionary D purely for signal reconstruction:
    min_{D, X}  ||Y - DX||_F^2     s.t. ||x_i||_0 <= T

LC-KSVD extends this by stacking two extra supervised terms into the
same matrix equation so that the original K-SVD solver can handle it
unchanged:

LC-KSVD1 adds a "discriminative sparse-code error":
    min_{D, A, X}  ||Y - DX||_F^2  +  alpha * ||Q - AX||_F^2
                   s.t. ||x_i||_0 <= T

    Q  : ideal "discriminative" sparse codes — binary matrix whose
         (j, i)-th entry is 1 iff dictionary atom j and signal y_i
         share the same class label.
    A  : linear transform that maps sparse codes toward Q.

LC-KSVD2 additionally adds a classification error:
    min_{D, W, A, X}  ||Y - DX||_F^2
                    + alpha * ||Q - AX||_F^2
                    + beta  * ||H - WX||_F^2
                   s.t. ||x_i||_0 <= T

    H  : one-hot label matrix  (num_classes x N).
    W  : linear classifier weights  (num_classes x K).

The key trick (Section 3.3 in [2]) is that by stacking the matrices:

    Y_new = [Y; sqrt(alpha)*Q; sqrt(beta)*H]
    D_new = [D; sqrt(alpha)*A; sqrt(beta)*W]

the combined objective becomes the standard K-SVD objective on Y_new
and D_new with the SAME sparse codes X:

    min_{D_new, X}  ||Y_new - D_new * X||_F^2    s.t. ||x_i||_0 <= T

After running K-SVD on these augmented matrices, D, A, W are extracted
from D_new and renormalised (Eq. 23 / Eq. 15 in [2]).

Sparse coding uses Orthogonal Matching Pursuit (OMP).

Dictionary update uses truncated SVD (one atom at a time, K-SVD style).

Attributes recovered after training
------------------------------------
D_hat : (n, K)          - dictionary atoms (L2-normalised per column)
A_hat : (K, K)          - linear transform for discriminative codes
W_hat : (num_classes, K) - linear classifier weights
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from numpy.linalg import norm, svd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Orthogonal Matching Pursuit (OMP)
# ---------------------------------------------------------------------------

def omp(y: np.ndarray, D: np.ndarray, sparsity: int) -> np.ndarray:
    """
    Orthogonal Matching Pursuit for a single signal.

    Solves:  argmin_x  ||y - Dx||_2^2    s.t.  ||x||_0 <= sparsity

    Parameters
    ----------
    y        : (n,)   signal to represent
    D        : (n, K) dictionary — columns should be L2-normalised
    sparsity : int    maximum number of nonzero coefficients

    Returns
    -------
    x : (K,) sparse coefficient vector
    """
    n, K = D.shape
    x = np.zeros(K)
    residual = y.copy()
    support: list[int] = []

    for _ in range(sparsity):
        # Correlate residual with all atoms
        correlations = np.abs(D.T @ residual)
        # Pick the most correlated atom not already in the support
        correlations[support] = -np.inf
        best = int(np.argmax(correlations))
        support.append(best)

        # Least-squares projection onto the current support (normal equations)
        D_s = D[:, support]  # (n, |support|)
        # x_s = (D_s^T D_s)^{-1} D_s^T y  — use lstsq for numerical stability
        x_s, _, _, _ = np.linalg.lstsq(D_s, y, rcond=None)

        # Update residual
        residual = y - D_s @ x_s

    # Place coefficients back into the full K-dimensional vector
    x[support] = x_s  # type: ignore[name-defined]
    return x


def batch_omp(Y: np.ndarray, D: np.ndarray, sparsity: int) -> np.ndarray:
    """
    OMP applied column-wise to all N signals in Y.

    Parameters
    ----------
    Y        : (n, N)  matrix of signals
    D        : (n, K)  dictionary
    sparsity : int     sparsity level T

    Returns
    -------
    X : (K, N) sparse code matrix
    """
    K = D.shape[1]
    N = Y.shape[1]
    X = np.zeros((K, N))
    for i in range(N):
        X[:, i] = omp(Y[:, i], D, sparsity)
    return X


# ---------------------------------------------------------------------------
# K-SVD dictionary update
# ---------------------------------------------------------------------------

def ksvd_update(
    Y: np.ndarray,
    D: np.ndarray,
    X: np.ndarray,
    sparsity: int,
    n_iter: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run K-SVD iterations on Y ≈ D @ X.

    Each atom d_k and its nonzero coefficients x_k_R are updated via a
    rank-1 SVD of the partial residual matrix E_k (Eq. 10-11 in [1]).

    Parameters
    ----------
    Y        : (m, N)  signal matrix (may be the stacked Y_new)
    D        : (m, K)  initial dictionary (L2-normalised columns)
    X        : (K, N)  initial sparse codes
    sparsity : int     sparsity level T
    n_iter   : int     number of outer K-SVD iterations

    Returns
    -------
    D : (m, K)  updated dictionary
    X : (K, N)  updated sparse codes
    """
    K = D.shape[1]

    for iteration in range(n_iter):
        # ---- Sparse coding step (OMP) ------------------------------------
        # Normalise dictionary columns before OMP so inner products are
        # proper cosine similarities.
        col_norms = norm(D, axis=0, keepdims=True)
        col_norms[col_norms == 0] = 1.0
        D_normed = D / col_norms
        X = batch_omp(Y, D_normed, sparsity)

        # ---- Dictionary update step (K-SVD) ------------------------------
        for k in range(K):
            # Indices of signals that use atom k
            omega = np.nonzero(X[k, :])[0]
            if omega.size == 0:
                # Atom is unused — reinitialise with the signal with the
                # largest current reconstruction error (common heuristic)
                residuals = Y - D @ X
                err_norms = norm(residuals, axis=0)
                worst = int(np.argmax(err_norms))
                new_atom = Y[:, worst] - D @ X[:, worst]
                n = norm(new_atom)
                D[:, k] = new_atom / n if n > 1e-10 else new_atom
                continue

            # Partial residual: error of all OTHER atoms for the signals
            # that involve atom k
            X_k_row = X[k, :].copy()
            X[k, :] = 0.0                           # temporarily zero out row k
            E_k = Y[:, omega] - D @ X[:, omega]    # (m, |omega|)
            X[k, :] = X_k_row                      # restore

            # Rank-1 SVD of E_k
            # E_k ≈ d_k * x_k_R^T  (best rank-1 approximation)
            U, s, Vt = svd(E_k, full_matrices=False)
            D[:, k] = U[:, 0]                       # updated atom
            X[k, omega] = s[0] * Vt[0, :]          # updated coefficients

        logger.debug("K-SVD iter %d / %d done", iteration + 1, n_iter)

    return D, X


# ---------------------------------------------------------------------------
# Label / discriminative-code helpers
# ---------------------------------------------------------------------------

def build_label_matrix(labels: np.ndarray, num_classes: int) -> np.ndarray:
    """
    Build one-hot label matrix H ∈ {0,1}^{num_classes × N}.

    Parameters
    ----------
    labels      : (N,) integer class indices, 0-based
    num_classes : int

    Returns
    -------
    H : (num_classes, N)
    """
    N = labels.shape[0]
    H = np.zeros((num_classes, N))
    H[labels, np.arange(N)] = 1.0
    return H


def build_discriminative_codes(
    labels: np.ndarray,
    atom_labels: np.ndarray,
) -> np.ndarray:
    """
    Build the discriminative sparse-code target matrix Q ∈ {0,1}^{K × N}.

    Q[:, i] is 1 at position k iff atom k and signal y_i share the same
    class label.  This is the core of the label consistency constraint
    (Section 3.1 in [1]).

    Parameters
    ----------
    labels      : (N,) class index for each training signal
    atom_labels : (K,) class index for each dictionary atom

    Returns
    -------
    Q : (K, N)
    """
    K = atom_labels.shape[0]
    N = labels.shape[0]
    Q = np.zeros((K, N))
    for k in range(K):
        match = labels == atom_labels[k]  # boolean mask over signals
        Q[k, match] = 1.0
    return Q


def init_atom_labels(
    labels: np.ndarray,
    num_classes: int,
    K: int,
) -> np.ndarray:
    """
    Assign a class label to each dictionary atom uniformly.

    Atoms are distributed across classes as evenly as possible with the
    remainder assigned to earlier classes (Section 3.3.1 in [1]).

    Parameters
    ----------
    labels      : unused here, kept for API consistency
    num_classes : int
    K           : total number of dictionary atoms

    Returns
    -------
    atom_labels : (K,) integer class indices
    """
    atoms_per_class = K // num_classes
    remainder = K % num_classes
    atom_labels = []
    for c in range(num_classes):
        count = atoms_per_class + (1 if c < remainder else 0)
        atom_labels.extend([c] * count)
    return np.array(atom_labels, dtype=int)


# ---------------------------------------------------------------------------
# Dictionary initialisation
# ---------------------------------------------------------------------------

def init_dictionary_ksvd(
    Y: np.ndarray,
    labels: np.ndarray,
    atom_labels: np.ndarray,
    sparsity: int,
    n_iter: int = 5,
) -> np.ndarray:
    """
    Initialise D^(0) by running a few iterations of standard K-SVD
    class-by-class and concatenating the per-class dictionaries
    (Section 3.3.1 in [1]).

    Parameters
    ----------
    Y           : (n, N)  all training signals
    labels      : (N,)    class index per signal
    atom_labels : (K,)    class index per atom
    sparsity    : int
    n_iter      : int     K-SVD iterations per class

    Returns
    -------
    D0 : (n, K)  initial dictionary
    """
    n = Y.shape[0]
    K = atom_labels.shape[0]
    D0 = np.zeros((n, K))
    num_classes = int(atom_labels.max()) + 1

    for c in range(num_classes):
        # Signals and atoms belonging to class c
        signal_mask = labels == c
        atom_mask = atom_labels == c
        Y_c = Y[:, signal_mask]                      # (n, N_c)
        K_c = int(atom_mask.sum())

        if Y_c.shape[1] == 0 or K_c == 0:
            continue

        # Initialise class-specific dictionary with random unit columns
        rng = np.random.default_rng(seed=c)
        D_c = rng.standard_normal((n, K_c))
        col_norms = norm(D_c, axis=0, keepdims=True)
        col_norms[col_norms == 0] = 1.0
        D_c /= col_norms

        X_c = np.zeros((K_c, Y_c.shape[1]))

        # Run a few K-SVD iterations on the class subset
        D_c, _ = ksvd_update(Y_c, D_c, X_c, sparsity, n_iter)

        D0[:, atom_mask] = D_c

    return D0


# ---------------------------------------------------------------------------
# Ridge regression helpers (Eq. 16 & 17 in [2])
# ---------------------------------------------------------------------------

def ridge_regression(X: np.ndarray, T: np.ndarray, lam: float) -> np.ndarray:
    """
    Solve:  argmin_W  ||T - WX||_F^2  + lam * ||W||_F^2

    Closed-form solution:  W = T @ X^T @ (X @ X^T + lam * I)^{-1}

    Parameters
    ----------
    X   : (K, N)     sparse codes
    T   : (m, N)     target matrix (Q or H)
    lam : float      regularisation parameter

    Returns
    -------
    W : (m, K)
    """
    K = X.shape[0]
    gram = X @ X.T + lam * np.eye(K)      # (K, K)
    return T @ X.T @ np.linalg.inv(gram)  # (m, K)


# ---------------------------------------------------------------------------
# Extract D, A, W from the joint D_new and renormalise (Eq. 23 / Eq. 15)
# ---------------------------------------------------------------------------

def extract_and_renorm(
    D_new: np.ndarray,
    n: int,
    K: int,
    alpha: float,
    beta: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    D_new has been column-normalised jointly, so each column k satisfies:

        ||[d_k; sqrt(alpha)*a_k; sqrt(beta)*w_k]||_2 = 1

    Recover the original-scale d_k, a_k, w_k by dividing by ||d_k||_2
    (the norm of just the top-n block of D_new[:, k]).

    Reference: Eq. 15 in [1] / Eq. 23 in [2].

    Parameters
    ----------
    D_new : (n + K + num_classes, K_total)  joint dictionary after K-SVD
    n     : feature dimensionality
    K     : number of dictionary atoms

    Returns
    -------
    D_hat : (n, K)
    A_hat : (K, K)
    W_hat : (num_classes, K)
    """
    D_block = D_new[:n, :]                          # (n, K)
    A_block = D_new[n:n + K, :] / np.sqrt(alpha)   # (K, K)
    W_block = D_new[n + K:, :] / np.sqrt(beta)     # (num_classes, K)

    # Per-column norm of the raw D block — used as the scale factor
    d_norms = norm(D_block, axis=0)                 # (K,)
    d_norms[d_norms == 0] = 1.0                     # guard against zero atom

    D_hat = D_block / d_norms                       # renormalised dictionary
    A_hat = A_block / d_norms                       # renormalised transform
    W_hat = W_block / d_norms                       # renormalised classifier
    return D_hat, A_hat, W_hat


# ---------------------------------------------------------------------------
# Main LC-KSVD class
# ---------------------------------------------------------------------------

@dataclass
class LCKSVDConfig:
    """
    Hyperparameters for LC-KSVD.

    Attributes
    ----------
    K        : number of dictionary atoms (total, across all classes)
    sparsity : sparsity level T — each signal uses at most T atoms
    n_iter   : number of outer LC-KSVD iterations (default 10 is often enough)
    alpha    : weight of the discriminative sparse-code error term
    beta     : weight of the classification error term
               Set beta=0 to use LC-KSVD1 (no direct classifier term in
               the joint objective); the classifier W is then fitted
               separately after convergence.
    lambda1  : ridge regularisation for W initialisation
    lambda2  : ridge regularisation for A initialisation
    init_iter: K-SVD iterations used for dictionary initialisation
    variant  : 'lcksvd1' or 'lcksvd2' — controls whether beta is used
               in the joint optimisation or only post-hoc.
    """
    K: int = 256
    sparsity: int = 30
    n_iter: int = 10
    alpha: float = 16.0      # recommended in [1] for face datasets
    beta: float = 4.0        # recommended in [1]
    lambda1: float = 1e-4    # regularisation for W
    lambda2: float = 1e-4    # regularisation for A
    init_iter: int = 5
    variant: str = "lcksvd2"  # 'lcksvd1' or 'lcksvd2'


class LCKSVD:
    """
    Label Consistent K-SVD (LC-KSVD).

    Learns a discriminative dictionary D, a linear transform A, and a
    linear classifier W simultaneously from labelled training signals Y.

    Usage
    -----
    >>> model = LCKSVD(LCKSVDConfig(K=256, sparsity=30, variant='lcksvd2'))
    >>> model.fit(Y_train, labels_train, num_classes=10)
    >>> predictions = model.predict(Y_test)
    """

    def __init__(self, config: LCKSVDConfig = LCKSVDConfig()) -> None:
        self.cfg = config
        # Learned parameters (set after .fit())
        self.D_hat: Optional[np.ndarray] = None
        self.A_hat: Optional[np.ndarray] = None
        self.W_hat: Optional[np.ndarray] = None
        self.atom_labels_: Optional[np.ndarray] = None
        self.num_classes_: Optional[int] = None

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(
        self,
        Y: np.ndarray,
        labels: np.ndarray,
        num_classes: int,
    ) -> "LCKSVD":
        """
        Learn dictionary D, transform A, and classifier W from labelled data.

        Parameters
        ----------
        Y           : (n, N)  training signal matrix, each column one sample
        labels      : (N,)    integer class indices in [0, num_classes)
        num_classes : int     number of classes

        Returns
        -------
        self
        """
        cfg = self.cfg
        n, N = Y.shape
        K = cfg.K
        self.num_classes_ = num_classes

        logger.info(
            "LC-KSVD fit: n=%d, N=%d, K=%d, sparsity=%d, variant=%s",
            n, N, K, cfg.sparsity, cfg.variant,
        )

        # ---- Step 1: Assign class labels to atoms -----------------------
        # Each atom is permanently assigned a class label; this label is
        # fixed throughout training (Section 3.3.1 in [2]).
        atom_labels = init_atom_labels(labels, num_classes, K)
        self.atom_labels_ = atom_labels

        # ---- Step 2: Build supervised targets ---------------------------
        # H: one-hot class label matrix  (num_classes, N)
        H = build_label_matrix(labels, num_classes)

        # Q: discriminative target sparse codes  (K, N)
        # Q[k, i] = 1 iff atom k and signal y_i share the same class.
        Q = build_discriminative_codes(labels, atom_labels)

        # ---- Step 3: Initialise D^(0) -----------------------------------
        logger.info("Initialising dictionary via per-class K-SVD ...")
        D0 = init_dictionary_ksvd(Y, labels, atom_labels, cfg.sparsity, cfg.init_iter)

        # ---- Step 4: Initialise sparse codes X^(0) ----------------------
        col_norms = norm(D0, axis=0, keepdims=True)
        col_norms[col_norms == 0] = 1.0
        D_normed = D0 / col_norms
        X0 = batch_omp(Y, D_normed, cfg.sparsity)

        # ---- Step 5: Initialise A^(0) and W^(0) via ridge regression ----
        # Eq. 16 in [2]: A = Q X^T (X X^T + lambda2 I)^{-1}
        A0 = ridge_regression(X0, Q, cfg.lambda2)

        # Eq. 17 in [2]: W = H X^T (X X^T + lambda1 I)^{-1}
        W0 = ridge_regression(X0, H, cfg.lambda1)

        # ---- Step 6: Build augmented (stacked) matrices -----------------
        # Y_new = [Y; sqrt(alpha)*Q; sqrt(beta)*H]   (Eq. 11 in [2])
        # D_new = [D; sqrt(alpha)*A; sqrt(beta)*W]
        #
        # For LC-KSVD1 we set beta=0 in the stacking so the classifier
        # term is excluded from the joint optimisation.
        if cfg.variant == "lcksvd1":
            effective_beta = 0.0
        else:
            effective_beta = cfg.beta

        Y_new = np.vstack([
            Y,
            np.sqrt(cfg.alpha) * Q,
            np.sqrt(effective_beta) * H,
        ])  # (n + K + num_classes, N)

        D_new = np.vstack([
            D0,
            np.sqrt(cfg.alpha) * A0,
            np.sqrt(effective_beta) * W0,
        ])  # (n + K + num_classes, K)

        # L2-normalise columns of D_new (required before K-SVD)
        col_norms = norm(D_new, axis=0, keepdims=True)
        col_norms[col_norms == 0] = 1.0
        D_new /= col_norms

        # ---- Step 7: Run K-SVD on the augmented system ------------------
        logger.info("Running K-SVD on augmented system for %d iterations ...", cfg.n_iter)
        D_new, X = ksvd_update(Y_new, D_new, X0, cfg.sparsity, cfg.n_iter)

        # ---- Step 8: Extract and renormalise D, A, W --------------------
        # Eq. 15 in [1] / Eq. 23 in [2]
        D_hat, A_hat, W_hat = extract_and_renorm(
            D_new, n, K, cfg.alpha, effective_beta if effective_beta > 0 else 1.0
        )

        # ---- Step 9: For LC-KSVD1, refit W separately ------------------
        # After D and A are learned, compute final sparse codes and fit W.
        if cfg.variant == "lcksvd1":
            logger.info("LC-KSVD1: fitting classifier W separately ...")
            X_final = batch_omp(Y, D_hat, cfg.sparsity)
            W_hat = ridge_regression(X_final, H, cfg.lambda1)

        self.D_hat = D_hat
        self.A_hat = A_hat
        self.W_hat = W_hat

        logger.info("LC-KSVD training complete.")
        return self

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def encode(self, Y: np.ndarray) -> np.ndarray:
        """
        Compute sparse codes for input signals using the learned dictionary.

        Parameters
        ----------
        Y : (n, N_test) test signals

        Returns
        -------
        X : (K, N_test) sparse codes
        """
        if self.D_hat is None:
            raise RuntimeError("Model is not fitted. Call .fit() first.")
        return batch_omp(Y, self.D_hat, self.cfg.sparsity)

    def predict(self, Y: np.ndarray) -> np.ndarray:
        """
        Classify test signals.

        Classification follows Eq. 16-17 in [1]:
          1. Compute sparse codes  x_i = OMP(y_i, D_hat, T)
          2. Classify via          label = argmax(W_hat @ x_i)

        Parameters
        ----------
        Y : (n, N_test) test signal matrix

        Returns
        -------
        predicted_labels : (N_test,) integer class indices
        """
        X = self.encode(Y)
        scores = self.W_hat @ X    # (num_classes, N_test)
        return np.argmax(scores, axis=0)

    def predict_scores(self, Y: np.ndarray) -> np.ndarray:
        """
        Return raw classifier scores (before argmax).

        Parameters
        ----------
        Y : (n, N_test)

        Returns
        -------
        scores : (num_classes, N_test)
        """
        X = self.encode(Y)
        return self.W_hat @ X


# ---------------------------------------------------------------------------
# Quick sanity-check / demonstration
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    rng = np.random.default_rng(seed=42)

    # --- Synthetic data: 3 classes, 50-d features, 20 atoms per class ------
    num_classes = 3
    n_features = 50
    n_train_per_class = 40
    n_test_per_class = 10
    K = num_classes * 20   # 60 total atoms (20 per class)
    sparsity = 6

    # Generate class-specific signal clusters
    centres = rng.standard_normal((num_classes, n_features))
    Y_train_list, Y_test_list, lbl_train_list, lbl_test_list = [], [], [], []
    for c in range(num_classes):
        Y_train_list.append(
            centres[c, :, None] + 0.1 * rng.standard_normal((n_features, n_train_per_class))
        )
        Y_test_list.append(
            centres[c, :, None] + 0.1 * rng.standard_normal((n_features, n_test_per_class))
        )
        lbl_train_list.extend([c] * n_train_per_class)
        lbl_test_list.extend([c] * n_test_per_class)

    Y_train = np.hstack(Y_train_list)                        # (50, 120)
    Y_test  = np.hstack(Y_test_list)                         # (50,  30)
    labels_train = np.array(lbl_train_list)                  # (120,)
    labels_test  = np.array(lbl_test_list)                   # ( 30,)

    # --- Train LC-KSVD2 ---------------------------------------------------
    cfg = LCKSVDConfig(
        K=K,
        sparsity=sparsity,
        n_iter=10,
        alpha=16.0,
        beta=4.0,
        variant="lcksvd2",
    )
    model = LCKSVD(cfg)
    model.fit(Y_train, labels_train, num_classes=num_classes)

    # --- Evaluate ---------------------------------------------------------
    preds = model.predict(Y_test)
    accuracy = float(np.mean(preds == labels_test))
    print(f"\nLC-KSVD2 test accuracy on synthetic data: {accuracy * 100:.1f}%")

    # --- Also try LC-KSVD1 ------------------------------------------------
    cfg1 = LCKSVDConfig(
        K=K,
        sparsity=sparsity,
        n_iter=10,
        alpha=16.0,
        variant="lcksvd1",
    )
    model1 = LCKSVD(cfg1)
    model1.fit(Y_train, labels_train, num_classes=num_classes)
    preds1 = model1.predict(Y_test)
    accuracy1 = float(np.mean(preds1 == labels_test))
    print(f"LC-KSVD1 test accuracy on synthetic data: {accuracy1 * 100:.1f}%")
