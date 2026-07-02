"""
LC-KSVD: Label Consistent K-SVD  —  GPU-accelerated (PyTorch)
==============================================================
Implementation based on:
  [1] Jiang, Z., Lin, Z., Davis, L.S. (2011).
      "Learning a Discriminative Dictionary for Sparse Coding via Label Consistent K-SVD."
      CVPR 2011, pp. 1697-1704.

  [2] Jiang, Z., Lin, Z., Davis, L.S. (2013).
      "Label Consistent K-SVD: Learning a Discriminative Dictionary for Recognition."
      IEEE TPAMI, 35(11), pp. 2651-2664.

GPU acceleration strategy
--------------------------
This file is a PyTorch-based GPU-accelerated version of lc_ksvd.py.

ALGORITHM LOGIC IS IDENTICAL — nothing in the mathematical steps has
changed. Only the array backend changes:

  CPU version (lc_ksvd.py)     → numpy arrays
  GPU version (this file)       → torch.Tensor on CUDA device

Why PyTorch instead of CuPy
-----------------------------
PyTorch is likely already installed in your environment (it is a standard
deep learning dependency). CuPy requires a separate installation that must
match your exact CUDA version. PyTorch manages its own CUDA runtime and
is more robust across CUDA versions.

Key PyTorch vs NumPy / CuPy differences handled here
------------------------------------------------------
1.  No module-swap pattern (xp=np / xp=cp).
    PyTorch uses torch.* functions + a device object.
    Every tensor is created on or moved to `self.device`.

2.  torch.linalg.lstsq returns a named tuple (.solution),
    not a 4-tuple like numpy. Handled in omp().

3.  Tensor indexing with a list must use torch.tensor(support, device=...).
    Plain Python lists cannot index a CUDA tensor directly.

4.  torch.nonzero(x, as_tuple=True) returns a tuple of 1D tensors.

5.  .item() converts a 0-d tensor to a Python scalar.

6.  All results are returned as CPU numpy arrays at the transfer boundary
    (fit / encode), so the rest of the pipeline is completely unaffected.

Transfer boundaries
--------------------
  fit(Y, labels)  : numpy in → GPU tensors during compute → numpy stored
  encode(Y)       : numpy in → GPU tensor → numpy out
  predict / predict_scores: call encode() then CPU matmul (W_hat is small)

Requirements
------------
  pip install torch          (CPU-only build)
  pip install torch --index-url https://download.pytorch.org/whl/cu118
                             (CUDA 11.8 build — adjust version as needed)

  Verify GPU:  python -c "import torch; print(torch.cuda.is_available())"
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)

# Global device — resolved once at import time
_CUDA_AVAILABLE = torch.cuda.is_available()


def _resolve_device(use_gpu: bool) -> torch.device:
    """
    Resolve the torch.device to use.

    Parameters
    ----------
    use_gpu : bool — whether GPU was requested

    Returns
    -------
    torch.device — cuda:0 if GPU is available and requested, else cpu
    """
    if use_gpu and _CUDA_AVAILABLE:
        return torch.device("cuda:0")
    if use_gpu and not _CUDA_AVAILABLE:
        logger.warning(
            "GPU requested but CUDA is not available. Falling back to CPU."
        )
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Transfer helpers
# ---------------------------------------------------------------------------

def _to_device(arr: np.ndarray, device: torch.device) -> torch.Tensor:
    """
    Convert a numpy array to a float64 torch.Tensor on the target device.

    float64 is used throughout to match numpy's default precision and
    avoid numerical differences relative to the CPU reference implementation.
    """
    return torch.from_numpy(arr.astype(np.float64)).to(device)


def _to_numpy(t: torch.Tensor) -> np.ndarray:
    """
    Move a torch.Tensor to CPU and convert to numpy array.

    Works whether the tensor is on CPU or GPU.
    """
    return t.detach().cpu().numpy()


# ---------------------------------------------------------------------------
# Orthogonal Matching Pursuit (OMP)
# ---------------------------------------------------------------------------

def _lstsq_gpu(A: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Solve the least-squares problem  argmin_x ||b - Ax||_2  on any device.

    torch.linalg.lstsq only supports driver='gels' on CUDA, which requires
    full column rank — unsafe for general use. Instead, we solve via the
    normal equations:

        x = (A^T A + eps*I)^{-1} A^T b

    The small Tikhonov term (eps=1e-10) keeps the Gram matrix non-singular
    when columns are nearly linearly dependent, which can happen in OMP when
    the support set is large. This matches the numerical behaviour of the
    CPU lstsq used in the original implementation and works identically on
    both CPU and CUDA tensors.

    Parameters
    ----------
    A : (n, s) tensor  — submatrix of dictionary atoms at current support
    b : (n,)   tensor  — signal or residual

    Returns
    -------
    x : (s,) tensor  — least-squares solution
    """
    eps = 1e-10
    gram = A.T @ A + eps * torch.eye(A.shape[1], dtype=A.dtype, device=A.device)
    rhs  = A.T @ b
    # torch.linalg.solve is supported on CUDA and is more stable than inv()
    return torch.linalg.solve(gram, rhs)


def omp(
    y: torch.Tensor,
    D: torch.Tensor,
    sparsity: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Orthogonal Matching Pursuit for a single signal.

    Solves:  argmin_x  ||y - Dx||_2^2    s.t.  ||x||_0 <= sparsity

    All operations run on `device` (GPU or CPU).

    Least-squares sub-problem solved via normal equations (_lstsq_gpu)
    instead of torch.linalg.lstsq because lstsq only supports driver='gels'
    on CUDA, which is not safe for all support set sizes.

    Parameters
    ----------
    y        : (n,)   signal tensor on `device`
    D        : (n, K) dictionary tensor on `device`
    sparsity : int    maximum number of nonzero coefficients
    device   : torch.device

    Returns
    -------
    x : (K,) sparse coefficient tensor on `device`
    """
    K = D.shape[1]
    x = torch.zeros(K, dtype=torch.float64, device=device)
    residual = y.clone()
    support: list[int] = []

    for _ in range(sparsity):
        # Correlate residual with all atoms — matmul on GPU
        correlations = torch.abs(D.T @ residual)          # (K,)

        # Mask already-selected atoms by setting their correlation to -inf
        if support:
            support_t = torch.tensor(support, dtype=torch.long, device=device)
            correlations[support_t] = -torch.inf

        # .item() pulls a Python int from the GPU tensor efficiently
        best = int(correlations.argmax().item())
        support.append(best)

        # Least-squares projection onto current support via normal equations
        support_t = torch.tensor(support, dtype=torch.long, device=device)
        D_s = D[:, support_t]                             # (n, |support|)
        x_s = _lstsq_gpu(D_s, y)                         # (|support|,)

        # Update residual
        residual = y - D_s @ x_s

    # Scatter coefficients back into the full K-dimensional vector
    support_t = torch.tensor(support, dtype=torch.long, device=device)
    x[support_t] = x_s   # type: ignore[name-defined]
    return x


def batch_omp(
    Y: torch.Tensor,
    D: torch.Tensor,
    sparsity: int,
    device: torch.device,
) -> torch.Tensor:
    """
    OMP applied column-wise to all N signals in Y.

    The Python loop over N signals is unavoidable in OMP (each signal's
    support set is data-dependent). But every inner operation — matmul,
    lstsq — runs on the GPU.

    Parameters
    ----------
    Y        : (n, N)  tensor on `device`
    D        : (n, K)  dictionary tensor on `device`
    sparsity : int
    device   : torch.device

    Returns
    -------
    X : (K, N) sparse code tensor on `device`
    """
    K = D.shape[1]
    N = Y.shape[1]
    X = torch.zeros((K, N), dtype=torch.float64, device=device)
    for i in range(N):
        X[:, i] = omp(Y[:, i], D, sparsity, device)
    return X


# ---------------------------------------------------------------------------
# K-SVD dictionary update
# ---------------------------------------------------------------------------

def ksvd_update(
    Y: torch.Tensor,
    D: torch.Tensor,
    X: torch.Tensor,
    sparsity: int,
    n_iter: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Run K-SVD iterations on Y ≈ D @ X.

    Each atom d_k and its nonzero coefficients x_k_R are updated via a
    rank-1 SVD of the partial residual matrix E_k (Eq. 10-11 in [1]).

    All heavy operations (matmul, SVD, norm) run on `device`.

    Parameters
    ----------
    Y        : (m, N)  signal tensor
    D        : (m, K)  dictionary tensor
    X        : (K, N)  sparse code tensor
    sparsity : int
    n_iter   : int
    device   : torch.device

    Returns
    -------
    D : (m, K)  updated dictionary tensor
    X : (K, N)  updated sparse code tensor
    """
    K = D.shape[1]

    for iteration in range(n_iter):
        # ---- Sparse coding step (OMP) ------------------------------------
        col_norms = torch.linalg.norm(D, dim=0, keepdim=True)    # (1, K)
        col_norms[col_norms == 0] = 1.0
        D_normed = D / col_norms
        X = batch_omp(Y, D_normed, sparsity, device)

        # ---- Dictionary update step (K-SVD) ------------------------------
        for k in range(K):
            # as_tuple=True returns a tuple of 1D tensors — take [0]
            omega = torch.nonzero(X[k, :], as_tuple=True)[0]     # (nnz,)

            if omega.numel() == 0:
                # Unused atom — reinitialise with worst-reconstructed signal
                residuals  = Y - D @ X                            # (m, N)
                err_norms  = torch.linalg.norm(residuals, dim=0)  # (N,)
                worst      = int(err_norms.argmax().item())
                new_atom   = Y[:, worst] - D @ X[:, worst]
                n_val      = torch.linalg.norm(new_atom)
                D[:, k]    = new_atom / n_val if n_val.item() > 1e-10 else new_atom
                continue

            # Partial residual over signals that use atom k
            X_k_row    = X[k, :].clone()
            X[k, :]    = 0.0
            E_k        = Y[:, omega] - D @ X[:, omega]            # (m, |omega|)
            X[k, :]    = X_k_row

            # Rank-1 SVD — torch.linalg.svd dispatches to cuBLAS on GPU
            # full_matrices=False gives compact (economy) SVD
            U, s, Vh   = torch.linalg.svd(E_k, full_matrices=False)
            D[:, k]    = U[:, 0]                                  # updated atom
            X[k, omega] = s[0] * Vh[0, :]                        # updated codes

        logger.debug("K-SVD iter %d / %d done", iteration + 1, n_iter)

    return D, X


# ---------------------------------------------------------------------------
# Label / discriminative-code helpers  (CPU — label ops only, no matmul)
# ---------------------------------------------------------------------------

def build_label_matrix(labels: np.ndarray, num_classes: int) -> np.ndarray:
    """
    Build one-hot label matrix H ∈ {0,1}^{num_classes × N}. (CPU numpy)

    Kept on CPU — label indexing is trivial and H is small.
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
    Build discriminative target sparse codes Q ∈ {0,1}^{K × N}. (CPU numpy)

    Q[k, i] = 1 iff atom k and signal y_i share the same class label.
    Kept on CPU — boolean comparison loop is negligible cost.
    """
    K = atom_labels.shape[0]
    N = labels.shape[0]
    Q = np.zeros((K, N))
    for k in range(K):
        Q[k, labels == atom_labels[k]] = 1.0
    return Q


def init_atom_labels(
    labels: np.ndarray,
    num_classes: int,
    K: int,
) -> np.ndarray:
    """Assign a class label to each dictionary atom uniformly. (CPU numpy)"""
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
    Y: torch.Tensor,
    labels: np.ndarray,
    atom_labels: np.ndarray,
    sparsity: int,
    n_iter: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Initialise D^(0) by running per-class K-SVD on the GPU.

    Parameters
    ----------
    Y          : (n, N) tensor already on `device`
    labels     : (N,)   CPU numpy array of class indices
    atom_labels: (K,)   CPU numpy array of atom class assignments
    sparsity   : int
    n_iter     : int
    device     : torch.device

    Returns
    -------
    D0 : (n, K) tensor on `device`
    """
    n = Y.shape[0]
    K = atom_labels.shape[0]
    D0 = torch.zeros((n, K), dtype=torch.float64, device=device)
    num_classes = int(atom_labels.max()) + 1

    rng = np.random.default_rng(seed=0)

    for c in range(num_classes):
        signal_mask = labels == c
        atom_mask   = atom_labels == c
        Y_c = Y[:, signal_mask]                          # (n, N_c) on device
        K_c = int(atom_mask.sum())

        if Y_c.shape[1] == 0 or K_c == 0:
            continue

        # Initialise random unit columns — create on CPU then move to device
        D_c_np = rng.standard_normal((n, K_c))
        col_norms_np = np.linalg.norm(D_c_np, axis=0, keepdims=True)
        col_norms_np[col_norms_np == 0] = 1.0
        D_c_np /= col_norms_np
        D_c = _to_device(D_c_np, device)

        X_c = torch.zeros((K_c, Y_c.shape[1]), dtype=torch.float64, device=device)
        D_c, _ = ksvd_update(Y_c, D_c, X_c, sparsity, n_iter, device)

        # atom_mask is a numpy boolean array — convert to torch for indexing
        atom_mask_t = torch.from_numpy(atom_mask).to(device)
        D0[:, atom_mask_t] = D_c

    return D0


# ---------------------------------------------------------------------------
# Ridge regression  (pure linear algebra — fully on GPU)
# ---------------------------------------------------------------------------

def ridge_regression(
    X: torch.Tensor,
    T: torch.Tensor,
    lam: float,
    device: torch.device,
) -> torch.Tensor:
    """
    Solve:  argmin_W  ||T - WX||_F^2  + lam * ||W||_F^2

    Closed form:  W = T @ X^T @ (X @ X^T + lam * I)^{-1}

    Parameters
    ----------
    X      : (K, N) sparse codes on `device`
    T      : (m, N) target matrix on `device`
    lam    : float  ridge regularisation parameter
    device : torch.device

    Returns
    -------
    W : (m, K) tensor on `device`
    """
    K = X.shape[0]
    gram = X @ X.T + lam * torch.eye(K, dtype=torch.float64, device=device)
    return T @ X.T @ torch.linalg.inv(gram)


# ---------------------------------------------------------------------------
# Extract D, A, W from D_new and renormalise (Eq. 15 / Eq. 23)
# ---------------------------------------------------------------------------

def extract_and_renorm(
    D_new: torch.Tensor,
    n: int,
    K: int,
    alpha: float,
    beta: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Recover D_hat, A_hat, W_hat from the jointly normalised D_new.

    All tensor slicing and division run on `device`.
    """
    D_block = D_new[:n, :]
    A_block = D_new[n:n + K, :] / torch.tensor(alpha, dtype=torch.float64, device=device).sqrt()
    W_block = D_new[n + K:, :]  / torch.tensor(beta,  dtype=torch.float64, device=device).sqrt()

    d_norms = torch.linalg.norm(D_block, dim=0)      # (K,)
    d_norms[d_norms == 0] = 1.0

    D_hat = D_block / d_norms
    A_hat = A_block / d_norms
    W_hat = W_block / d_norms
    return D_hat, A_hat, W_hat


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class LCKSVDConfig:
    """
    Hyperparameters for LC-KSVD.

    Attributes
    ----------
    K        : number of dictionary atoms
    sparsity : sparsity level T
    n_iter   : outer LC-KSVD iterations
    alpha    : weight of discriminative sparse-code error term
    beta     : weight of classification error term (LC-KSVD2 only)
    lambda1  : ridge regularisation for W
    lambda2  : ridge regularisation for A
    init_iter: K-SVD iterations for dictionary initialisation
    variant  : 'lcksvd1' or 'lcksvd2'
    use_gpu  : if True, run on CUDA GPU via PyTorch.
               Falls back to CPU automatically if CUDA is unavailable.
    """
    K: int          = 128
    sparsity: int   = 30
    n_iter: int     = 10
    alpha: float    = 16.0
    beta: float     = 4.0
    lambda1: float  = 1e-4
    lambda2: float  = 1e-4
    init_iter: int  = 5
    variant: str    = "lcksvd2"
    use_gpu: bool   = True


# ---------------------------------------------------------------------------
# Main LC-KSVD class
# ---------------------------------------------------------------------------

class LCKSVD:
    """
    Label Consistent K-SVD (LC-KSVD) — PyTorch GPU backend.

    All public methods accept and return standard numpy arrays (CPU).
    PyTorch tensors are used internally and converted back to numpy at
    the transfer boundary — the rest of the pipeline is unaffected.

    Usage
    -----
    >>> cfg = LCKSVDConfig(K=256, sparsity=10, use_gpu=True)
    >>> model = LCKSVD(cfg)
    >>> model.fit(Y_train, labels_train, num_classes=2)
    >>> codes = model.encode(Y_test)       # numpy array
    >>> preds = model.predict(Y_test)      # numpy array
    """

    def __init__(self, config: LCKSVDConfig = LCKSVDConfig()) -> None:
        self.cfg    = config
        self.device = _resolve_device(config.use_gpu)
        self.on_gpu = self.device.type == "cuda"

        # Learned parameters — always stored as CPU numpy arrays after fit()
        self.D_hat: Optional[np.ndarray] = None
        self.A_hat: Optional[np.ndarray] = None
        self.W_hat: Optional[np.ndarray] = None
        self.atom_labels_: Optional[np.ndarray] = None
        self.num_classes_: Optional[int] = None

        logger.info(
            "LCKSVD initialised. Backend: %s",
            f"GPU ({self.device})" if self.on_gpu else "CPU (PyTorch)",
        )

    # ------------------------------------------------------------------
    # fit
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
        Y           : (n, N)  training signal matrix — CPU numpy array
        labels      : (N,)    0-based integer class indices — CPU numpy
        num_classes : int

        Returns
        -------
        self
        """
        cfg    = self.cfg
        device = self.device
        n, N   = Y.shape
        K      = cfg.K
        self.num_classes_ = num_classes

        logger.info(
            "LC-KSVD fit: n=%d, N=%d, K=%d, sparsity=%d, variant=%s, device=%s",
            n, N, K, cfg.sparsity, cfg.variant, device,
        )

        # ---- Transfer Y to device ----------------------------------------
        # Labels stay as numpy — they are only used for index arithmetic.
        Y_dev = _to_device(Y, device)                            # (n, N) on GPU

        # ---- Step 1: Assign class labels to atoms (CPU) ------------------
        atom_labels = init_atom_labels(labels, num_classes, K)
        self.atom_labels_ = atom_labels

        # ---- Step 2: Build supervised targets (CPU → GPU) ----------------
        H_cpu = build_label_matrix(labels, num_classes)          # (num_classes, N)
        Q_cpu = build_discriminative_codes(labels, atom_labels)  # (K, N)

        H = _to_device(H_cpu, device)
        Q = _to_device(Q_cpu, device)

        # ---- Step 3: Initialise D^(0) (GPU) ------------------------------
        logger.info("Initialising dictionary via per-class K-SVD ...")
        D0 = init_dictionary_ksvd(
            Y_dev, labels, atom_labels, cfg.sparsity, cfg.init_iter, device
        )

        # ---- Step 4: Initialise sparse codes X^(0) (GPU) -----------------
        col_norms = torch.linalg.norm(D0, dim=0, keepdim=True)
        col_norms[col_norms == 0] = 1.0
        D_normed = D0 / col_norms
        X0 = batch_omp(Y_dev, D_normed, cfg.sparsity, device)

        # ---- Step 5: Initialise A^(0) and W^(0) via ridge regression -----
        A0 = ridge_regression(X0, Q, cfg.lambda2, device)
        W0 = ridge_regression(X0, H, cfg.lambda1, device)

        # ---- Step 6: Build augmented matrices (GPU) ----------------------
        effective_beta = 0.0 if cfg.variant == "lcksvd1" else cfg.beta

        sq_alpha = torch.tensor(cfg.alpha,       dtype=torch.float64, device=device).sqrt()
        sq_beta  = torch.tensor(effective_beta,  dtype=torch.float64, device=device).sqrt()

        Y_new = torch.vstack([Y_dev, sq_alpha * Q, sq_beta * H])
        D_new = torch.vstack([D0,    sq_alpha * A0, sq_beta * W0])

        col_norms = torch.linalg.norm(D_new, dim=0, keepdim=True)
        col_norms[col_norms == 0] = 1.0
        D_new = D_new / col_norms

        # ---- Step 7: K-SVD on the augmented system (GPU) -----------------
        logger.info(
            "Running K-SVD on augmented system for %d iterations ...", cfg.n_iter
        )
        D_new, X = ksvd_update(Y_new, D_new, X0, cfg.sparsity, cfg.n_iter, device)

        # ---- Step 8: Extract and renormalise D, A, W (GPU) ---------------
        D_hat_dev, A_hat_dev, W_hat_dev = extract_and_renorm(
            D_new, n, K, cfg.alpha,
            effective_beta if effective_beta > 0 else 1.0,
            device,
        )

        # ---- Step 9: LC-KSVD1 — refit W separately (GPU) ----------------
        if cfg.variant == "lcksvd1":
            logger.info("LC-KSVD1: fitting classifier W separately ...")
            X_final   = batch_omp(Y_dev, D_hat_dev, cfg.sparsity, device)
            W_hat_dev = ridge_regression(X_final, H, cfg.lambda1, device)

        # ---- Transfer learned parameters back to CPU numpy ---------------
        # Downstream consumers (Evaluator, LCKSVDEvaluator, sklearn)
        # always receive plain numpy arrays.
        self.D_hat = _to_numpy(D_hat_dev)
        self.A_hat = _to_numpy(A_hat_dev)
        self.W_hat = _to_numpy(W_hat_dev)

        logger.info("LC-KSVD training complete.")
        return self

    # ------------------------------------------------------------------
    # encode
    # ------------------------------------------------------------------

    def encode(self, Y: np.ndarray) -> np.ndarray:
        """
        Compute sparse codes for input signals using the learned dictionary.

        Parameters
        ----------
        Y : (n, N_test) — CPU numpy array

        Returns
        -------
        X : (K, N_test) — CPU numpy array
        """
        if self.D_hat is None:
            raise RuntimeError("Model is not fitted. Call .fit() first.")

        device = self.device
        Y_dev  = _to_device(Y, device)
        D_dev  = _to_device(self.D_hat, device)

        X_dev = batch_omp(Y_dev, D_dev, self.cfg.sparsity, device)
        return _to_numpy(X_dev)

    # ------------------------------------------------------------------
    # predict
    # ------------------------------------------------------------------

    def predict(self, Y: np.ndarray) -> np.ndarray:
        """
        Classify test signals via the internal W_hat linear classifier.

        Parameters
        ----------
        Y : (n, N_test) — CPU numpy array

        Returns
        -------
        predicted_labels : (N_test,) — CPU numpy array of class indices
        """
        X      = self.encode(Y)            # CPU numpy (K, N_test)
        scores = self.W_hat @ X            # CPU matmul — W_hat is small
        return np.argmax(scores, axis=0)

    # ------------------------------------------------------------------
    # predict_scores
    # ------------------------------------------------------------------

    def predict_scores(self, Y: np.ndarray) -> np.ndarray:
        """
        Return raw classifier scores (before argmax).

        Parameters
        ----------
        Y : (n, N_test) — CPU numpy array

        Returns
        -------
        scores : (num_classes, N_test) — CPU numpy array
        """
        X = self.encode(Y)
        return self.W_hat @ X


# ---------------------------------------------------------------------------
# Quick sanity-check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    rng = np.random.default_rng(seed=42)

    num_classes        = 3
    n_features         = 50
    n_train_per_class  = 40
    n_test_per_class   = 10
    K                  = num_classes * 20
    sparsity           = 6

    centres = rng.standard_normal((num_classes, n_features))
    Y_train_list, Y_test_list = [], []
    lbl_train_list, lbl_test_list = [], []

    for c in range(num_classes):
        Y_train_list.append(
            centres[c, :, None] + 0.1 * rng.standard_normal((n_features, n_train_per_class))
        )
        Y_test_list.append(
            centres[c, :, None] + 0.1 * rng.standard_normal((n_features, n_test_per_class))
        )
        lbl_train_list.extend([c] * n_train_per_class)
        lbl_test_list.extend([c] * n_test_per_class)

    Y_train      = np.hstack(Y_train_list)
    Y_test       = np.hstack(Y_test_list)
    labels_train = np.array(lbl_train_list)
    labels_test  = np.array(lbl_test_list)

    cfg   = LCKSVDConfig(K=K, sparsity=sparsity, n_iter=10, use_gpu=True)
    model = LCKSVD(cfg)
    model.fit(Y_train, labels_train, num_classes=num_classes)

    preds = model.predict(Y_test)
    print(f"Test accuracy: {np.mean(preds == labels_test) * 100:.1f}%")