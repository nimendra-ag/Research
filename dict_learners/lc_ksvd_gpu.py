"""
LC-KSVD: Label Consistent K-SVD  —  PyTorch GPU variant
=========================================================
Implementation based on:
  [1] Jiang, Z., Lin, Z., Davis, L.S. (2011).
      "Learning a Discriminative Dictionary for Sparse Coding via Label Consistent K-SVD."
      CVPR 2011, pp. 1697-1704.

  [2] Jiang, Z., Lin, Z., Davis, L.S. (2013).
      "Label Consistent K-SVD: Learning a Discriminative Dictionary for Recognition."
      IEEE TPAMI, 35(11), pp. 2651-2664.

Relation to lc_ksvd.py (CPU version)
--------------------------------------
This file is the GPU-accelerated counterpart of lc_ksvd.py.
ZERO algorithmic logic has been changed. Every mathematical step,
every loop structure, every equation is identical.

The only changes are:
  1. Array backend: numpy arrays → torch.Tensor on CUDA device.
  2. Two CPU-bound operations (OMP, SVD init) temporarily move data
     to CPU numpy, call the CPU library, then move the result back
     to GPU. See "What stays on CPU" below.
  3. Transfer helpers (_to_device, _to_cpu) added at module level.
  4. LCKSVDConfig gains a use_gpu: bool field.
  5. LCKSVD.__init__ resolves a torch.device from use_gpu.

What runs on GPU (CUDA)
------------------------
  aksvd_update  — all matmuls (D.T@D, D.T@Y, D@X, R@g, R.T@d),
                  norm, boolean masking, column updates
  ridge_regression — matmul chain + torch.linalg.inv
  extract_and_renorm — slicing, division, norm
  fit steps 4, 6, 8, 9 — matmuls for gram/Xy/stacking/normalisation

What stays on CPU (cannot run on CUDA)
----------------------------------------
  orthogonal_mp_gram (sklearn) — no CUDA backend exists.
      Strategy: pull gram (K,K) and Xy (K,N) to CPU numpy,
      call orthogonal_mp_gram, push result (K,N) back to GPU.
      gram and Xy are small relative to Y and D, so the transfer
      cost is low. The heavy matmuls that produce gram and Xy
      still run on GPU.

  scipy.sparse.linalg.svds (init_dictionary_svd) — CPU only.
      Strategy: pull Y_c (n, N_c) to CPU for each class subset,
      call svds, push D_c result back to GPU.
      This happens once at initialisation, not per iteration.

  build_label_matrix, build_discriminative_codes, init_atom_labels
      — pure label index arithmetic, negligible cost, stay CPU.

Transfer boundary
------------------
  fit(Y, labels)   : numpy in → GPU tensor at entry
                     GPU tensors → numpy stored in self.D_hat/A_hat/W_hat
  encode(Y)        : numpy in → GPU tensor → numpy out
  predict/predict_scores : call encode() then CPU matmul (W_hat small)

The pipeline adapter (dict_learners/lcksvd.py), Evaluator, and
LCKSVDEvaluator need NO changes — they always see CPU numpy arrays.

Requirements
------------
  pip install torch --index-url https://download.pytorch.org/whl/cu118
  (adjust cu118 to match your CUDA version)

  Verify: python -c "import torch; print(torch.cuda.is_available())"
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import scipy.sparse.linalg
import torch
from sklearn.linear_model import orthogonal_mp_gram

logger = logging.getLogger(__name__)

_CUDA_AVAILABLE = torch.cuda.is_available()


# ---------------------------------------------------------------------------
# Transfer helpers
# ---------------------------------------------------------------------------

def _resolve_device(use_gpu: bool) -> torch.device:
    """Return cuda:0 if GPU requested and available, else cpu."""
    if use_gpu and _CUDA_AVAILABLE:
        return torch.device("cuda:0")
    if use_gpu and not _CUDA_AVAILABLE:
        logger.warning("GPU requested but CUDA unavailable — falling back to CPU.")
    return torch.device("cpu")


def _to_device(arr: np.ndarray, device: torch.device) -> torch.Tensor:
    """Convert a numpy array to a float64 torch.Tensor on device."""
    return torch.from_numpy(np.asarray(arr, dtype=np.float64)).to(device)


def _to_cpu(t: torch.Tensor) -> np.ndarray:
    """Move a tensor to CPU and return as numpy array."""
    return t.detach().cpu().numpy()


# ---------------------------------------------------------------------------
# Approximate K-SVD dictionary update
# ---------------------------------------------------------------------------

def aksvd_update(
    Y: torch.Tensor,
    D: torch.Tensor,
    X: torch.Tensor,
    sparsity: int,
    n_iter: int,
    tol: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Run Approximate K-SVD iterations on Y ≈ D @ X.

    Logic is identical to lc_ksvd.py:aksvd_update().
    All tensor operations run on `device`.

    The sparse coding step (orthogonal_mp_gram) temporarily transfers
    gram and Xy to CPU numpy, calls sklearn, and pushes X back to GPU.
    Every other operation in this function stays on GPU.

    Parameters  (all tensors on `device`)
    ----------
    Y        : (m, N)  signal or augmented matrix
    D        : (m, K)  dictionary — columns are atoms
    X        : (K, N)  sparse codes
    sparsity : int
    n_iter   : int
    tol      : float   early-stop threshold on ||Y - DX||_F
    device   : torch.device

    Returns   (both on `device`)
    -------
    D : (m, K)
    X : (K, N)
    """
    K = D.shape[1]

    for iteration in range(n_iter):
        # ---- Sparse coding (orthogonal_mp_gram via CPU) ------------------
        # [CHANGED vs CPU] matmuls run on GPU; gram/Xy pulled to numpy
        # only for the sklearn call, then X pushed back to GPU.
        col_norms = torch.linalg.norm(D, dim=0, keepdim=True)   # (1, K) on GPU
        col_norms[col_norms == 0] = 1.0
        D_normed = D / col_norms                                  # (m, K) on GPU

        gram_gpu = D_normed.T @ D_normed                         # (K, K) on GPU
        Xy_gpu   = D_normed.T @ Y                                # (K, N) on GPU

        # Pull to CPU numpy for sklearn — gram and Xy are small
        gram_cpu = _to_cpu(gram_gpu)                             # (K, K) numpy
        Xy_cpu   = _to_cpu(Xy_gpu)                              # (K, N) numpy

        X_cpu = orthogonal_mp_gram(                              # (K, N) numpy
            gram_cpu, Xy_cpu, n_nonzero_coefs=sparsity
        )
        X = _to_device(X_cpu, device)                           # (K, N) back on GPU

        # ---- Tolerance check -----------------------------------------------
        recon_err = torch.linalg.norm(Y - D_normed @ X).item()
        if recon_err < tol:
            logger.debug(
                "Early stop at iter %d / %d  (err=%.6f < tol=%.6f)",
                iteration + 1, n_iter, recon_err, tol,
            )
            D = D_normed
            break

        # ---- Dictionary update (power-method, fully on GPU) ---------------
        D = D_normed.clone()

        for k in range(K):
            # [CHANGED vs CPU] bool mask on GPU tensor
            I = X[k, :] != 0                                    # (N,) bool on GPU

            if not I.any():
                residuals = Y - D @ X                           # (m, N) on GPU
                err_norms = torch.linalg.norm(residuals, dim=0) # (N,)
                worst     = int(err_norms.argmax().item())
                new_atom  = Y[:, worst].clone()
                n_val     = torch.linalg.norm(new_atom).item()
                D[:, k]   = new_atom / n_val if n_val > 1e-10 else new_atom
                continue

            g = X[k, I]                                         # (nnz,) on GPU

            d_k_saved = D[:, k].clone()
            D[:, k]   = 0.0
            R         = Y[:, I] - D @ X[:, I]                  # (m, nnz) on GPU
            D[:, k]   = d_k_saved

            d_new = R @ g                                       # (m,) on GPU
            n_val = torch.linalg.norm(d_new).item()
            if n_val < 1e-10:
                continue
            d_new = d_new / n_val

            D[:, k] = d_new
            X[k, I] = R.T @ d_new                              # (nnz,) on GPU

        logger.debug("AKSVD iter %d / %d done", iteration + 1, n_iter)

    return D, X


# ---------------------------------------------------------------------------
# Label / discriminative-code helpers  (CPU only — unchanged logic)
# ---------------------------------------------------------------------------

def build_label_matrix(labels: np.ndarray, num_classes: int) -> np.ndarray:
    """One-hot label matrix H ∈ {0,1}^{num_classes × N}. (CPU numpy)"""
    N = labels.shape[0]
    H = np.zeros((num_classes, N))
    H[labels, np.arange(N)] = 1.0
    return H


def build_discriminative_codes(
    labels: np.ndarray,
    atom_labels: np.ndarray,
) -> np.ndarray:
    """Discriminative target Q ∈ {0,1}^{K × N}. (CPU numpy)"""
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
    """Assign class labels to dictionary atoms uniformly. (CPU numpy)"""
    atoms_per_class = K // num_classes
    remainder       = K % num_classes
    atom_labels     = []
    for c in range(num_classes):
        count = atoms_per_class + (1 if c < remainder else 0)
        atom_labels.extend([c] * count)
    return np.array(atom_labels, dtype=int)


# ---------------------------------------------------------------------------
# Dictionary initialisation  (SVD per class — CPU scipy, result pushed to GPU)
# ---------------------------------------------------------------------------

def init_dictionary_svd(
    Y: torch.Tensor,
    labels: np.ndarray,
    atom_labels: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    """
    Per-class SVD-based dictionary initialisation.

    Logic is identical to lc_ksvd.py:init_dictionary_svd().

    [CHANGED vs CPU] Y is a GPU tensor. Each class subset Y_c is pulled
    to CPU numpy for scipy.sparse.linalg.svds (CPU-only), then the
    resulting D_c is pushed back to the GPU tensor D0.
    This happens once at training start, not per iteration.

    Parameters
    ----------
    Y          : (n, N) tensor on `device`
    labels     : (N,)   CPU numpy array
    atom_labels: (K,)   CPU numpy array
    device     : torch.device

    Returns
    -------
    D0 : (n, K) tensor on `device`
    """
    n           = Y.shape[0]
    K           = atom_labels.shape[0]
    D0          = torch.zeros((n, K), dtype=torch.float64, device=device)
    num_classes = int(atom_labels.max()) + 1
    rng         = np.random.default_rng(seed=42)

    for c in range(num_classes):
        signal_mask = labels == c
        atom_mask   = atom_labels == c
        K_c         = int(atom_mask.sum())

        # [CHANGED vs CPU] Pull class subset to CPU for scipy svds
        Y_c_cpu = _to_cpu(Y[:, signal_mask])                # (n, N_c) numpy

        if Y_c_cpu.shape[1] == 0 or K_c == 0:
            continue

        if Y_c_cpu.shape[1] < K_c:
            D_c_cpu = rng.standard_normal((n, K_c))
        else:
            try:
                _, s, vt = scipy.sparse.linalg.svds(Y_c_cpu.T, k=K_c)
                s        = s[::-1]
                vt       = vt[::-1, :]
                D_c_cpu  = (np.diag(s) @ vt).T              # (n, K_c) numpy
            except Exception:
                D_c_cpu = rng.standard_normal((n, K_c))

        col_norms = np.linalg.norm(D_c_cpu, axis=0, keepdims=True)
        col_norms[col_norms == 0] = 1.0
        D_c_cpu /= col_norms

        # Push result back to GPU
        D0[:, atom_mask] = _to_device(D_c_cpu, device)      # assign on GPU

    return D0


# ---------------------------------------------------------------------------
# Ridge regression  (fully on GPU)
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

    Logic identical to lc_ksvd.py. All ops on GPU.

    Parameters  (all on `device`)
    ----------
    X   : (K, N)
    T   : (m, N)
    lam : float

    Returns
    -------
    W : (m, K) on `device`
    """
    K    = X.shape[0]
    gram = X @ X.T + lam * torch.eye(K, dtype=torch.float64, device=device)
    return T @ X.T @ torch.linalg.inv(gram)


# ---------------------------------------------------------------------------
# Extract and renormalise  (fully on GPU)
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
    Recover D_hat, A_hat, W_hat from jointly normalised D_new.

    Logic identical to lc_ksvd.py. All ops on GPU.
    """
    D_block = D_new[:n, :]
    A_block = D_new[n:n + K, :] / torch.tensor(alpha, dtype=torch.float64, device=device).sqrt()
    W_block = D_new[n + K:, :]  / torch.tensor(beta,  dtype=torch.float64, device=device).sqrt()

    d_norms = torch.linalg.norm(D_block, dim=0)             # (K,)
    d_norms[d_norms == 0] = 1.0

    return D_block / d_norms, A_block / d_norms, W_block / d_norms


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class LCKSVDConfig:
    """
    Hyperparameters for LC-KSVD.

    Identical to lc_ksvd.py:LCKSVDConfig with one addition:
      use_gpu : bool — set True to run on CUDA GPU via PyTorch.
                       Falls back to CPU automatically if CUDA unavailable.
    """
    K: int         = 256
    sparsity: int  = 30
    n_iter: int    = 10
    tol: float     = 1e-6
    alpha: float   = 16.0
    beta: float    = 4.0
    lambda1: float = 1e-4
    lambda2: float = 1e-4
    variant: str   = "lcksvd2"
    use_gpu: bool  = True        # [ADDED vs CPU]


# ---------------------------------------------------------------------------
# Main LC-KSVD class
# ---------------------------------------------------------------------------

class LCKSVD:
    """
    Label Consistent K-SVD — Approximate K-SVD variant, PyTorch GPU backend.

    Public API is identical to lc_ksvd.py:LCKSVD.
    All public methods accept and return CPU numpy arrays.
    GPU tensors are used internally and converted back at the boundary.

    Usage
    -----
    >>> cfg   = LCKSVDConfig(K=256, sparsity=10, use_gpu=True)
    >>> model = LCKSVD(cfg)
    >>> model.fit(Y_train, labels_train, num_classes=2)
    >>> codes = model.encode(Y_test)    # numpy (K, N_test)
    >>> preds = model.predict(Y_test)   # numpy (N_test,)
    """

    def __init__(self, config: LCKSVDConfig = LCKSVDConfig()) -> None:
        self.cfg    = config
        self.device = _resolve_device(config.use_gpu)           # [ADDED vs CPU]
        self.on_gpu = self.device.type == "cuda"

        self.D_hat: Optional[np.ndarray] = None
        self.A_hat: Optional[np.ndarray] = None
        self.W_hat: Optional[np.ndarray] = None
        self.atom_labels_: Optional[np.ndarray] = None
        self.num_classes_: Optional[int] = None

        logger.info(
            "LCKSVD initialised — device: %s", self.device
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
        Learn dictionary D, transform A, and classifier W.

        Parameters
        ----------
        Y           : (n, N)  CPU numpy — moved to GPU at entry
        labels      : (N,)    CPU numpy — stays CPU throughout
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
            "LC-KSVD fit [AKSVD-GPU]: n=%d, N=%d, K=%d, sparsity=%d, variant=%s, device=%s",
            n, N, K, cfg.sparsity, cfg.variant, device,
        )

        # ---- [CHANGED] Transfer Y to GPU ---------------------------------
        Y_dev = _to_device(Y, device)                           # (n, N) on GPU

        # ---- Step 1: Assign class labels to atoms (CPU) ------------------
        atom_labels = init_atom_labels(labels, num_classes, K)
        self.atom_labels_ = atom_labels

        # ---- Step 2: Build supervised targets (CPU → GPU) ----------------
        H_cpu = build_label_matrix(labels, num_classes)         # (num_classes, N)
        Q_cpu = build_discriminative_codes(labels, atom_labels) # (K, N)
        H     = _to_device(H_cpu, device)                       # on GPU
        Q     = _to_device(Q_cpu, device)                       # on GPU

        # ---- Step 3: Initialise D^(0) via per-class SVD (CPU→GPU) -------
        logger.info("Initialising dictionary via per-class SVD ...")
        D0 = init_dictionary_svd(Y_dev, labels, atom_labels, device)  # (n, K) on GPU

        # ---- Step 4: Initialise sparse codes X^(0) -----------------------
        # [CHANGED] matmuls on GPU; gram/Xy pulled to CPU for sklearn OMP
        col_norms = torch.linalg.norm(D0, dim=0, keepdim=True)
        col_norms[col_norms == 0] = 1.0
        D0_normed  = D0 / col_norms
        gram_cpu   = _to_cpu(D0_normed.T @ D0_normed)           # (K, K) numpy
        Xy_cpu     = _to_cpu(D0_normed.T @ Y_dev)               # (K, N) numpy
        X0         = _to_device(                                 # (K, N) on GPU
            orthogonal_mp_gram(gram_cpu, Xy_cpu, n_nonzero_coefs=cfg.sparsity),
            device,
        )

        # ---- Step 5: Initialise A^(0) and W^(0) (GPU) -------------------
        A0 = ridge_regression(X0, Q, cfg.lambda2, device)
        W0 = ridge_regression(X0, H, cfg.lambda1, device)

        # ---- Step 6: Build augmented matrices (GPU) ----------------------
        effective_beta = 0.0 if cfg.variant == "lcksvd1" else cfg.beta

        sq_alpha = torch.tensor(cfg.alpha,      dtype=torch.float64, device=device).sqrt()
        sq_beta  = torch.tensor(effective_beta, dtype=torch.float64, device=device).sqrt()

        Y_new = torch.vstack([Y_dev, sq_alpha * Q,  sq_beta * H ])  # (n+K+C, N)
        D_new = torch.vstack([D0,    sq_alpha * A0, sq_beta * W0])  # (n+K+C, K)

        col_norms = torch.linalg.norm(D_new, dim=0, keepdim=True)
        col_norms[col_norms == 0] = 1.0
        D_new = D_new / col_norms

        # ---- Step 7: Approximate K-SVD on augmented system (GPU) ---------
        logger.info(
            "Running AKSVD on augmented system for up to %d iterations ...",
            cfg.n_iter,
        )
        D_new, X = aksvd_update(
            Y_new, D_new, X0, cfg.sparsity, cfg.n_iter, cfg.tol, device
        )

        # ---- Step 8: Extract and renormalise (GPU) -----------------------
        D_hat_dev, A_hat_dev, W_hat_dev = extract_and_renorm(
            D_new, n, K,
            cfg.alpha,
            effective_beta if effective_beta > 0 else 1.0,
            device,
        )

        # ---- Step 9: LC-KSVD1 — refit W separately (GPU + CPU OMP) ------
        if cfg.variant == "lcksvd1":
            logger.info("LC-KSVD1: fitting classifier W separately ...")
            col_norms    = torch.linalg.norm(D_hat_dev, dim=0, keepdim=True)
            col_norms[col_norms == 0] = 1.0
            D_hat_normed = D_hat_dev / col_norms
            gram_cpu     = _to_cpu(D_hat_normed.T @ D_hat_normed)   # (K, K)
            Xy_cpu       = _to_cpu(D_hat_normed.T @ Y_dev)          # (K, N)
            X_final      = _to_device(
                orthogonal_mp_gram(gram_cpu, Xy_cpu, n_nonzero_coefs=cfg.sparsity),
                device,
            )
            W_hat_dev = ridge_regression(X_final, H, cfg.lambda1, device)

        # ---- [CHANGED] Transfer learned params back to CPU numpy ---------
        self.D_hat = _to_cpu(D_hat_dev)
        self.A_hat = _to_cpu(A_hat_dev)
        self.W_hat = _to_cpu(W_hat_dev)

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
        Y : (n, N_test)  CPU numpy

        Returns
        -------
        X : (K, N_test)  CPU numpy
        """
        if self.D_hat is None:
            raise RuntimeError("Model is not fitted. Call .fit() first.")

        device    = self.device
        Y_dev     = _to_device(Y, device)
        D_dev     = _to_device(self.D_hat, device)

        col_norms = torch.linalg.norm(D_dev, dim=0, keepdim=True)
        col_norms[col_norms == 0] = 1.0
        D_normed  = D_dev / col_norms

        # [CHANGED] matmuls on GPU; pull to CPU for sklearn
        gram_cpu  = _to_cpu(D_normed.T @ D_normed)              # (K, K)
        Xy_cpu    = _to_cpu(D_normed.T @ Y_dev)                 # (K, N_test)
        return orthogonal_mp_gram(gram_cpu, Xy_cpu, n_nonzero_coefs=self.cfg.sparsity)

    # ------------------------------------------------------------------
    # predict / predict_scores  (unchanged logic)
    # ------------------------------------------------------------------

    def predict(self, Y: np.ndarray) -> np.ndarray:
        """
        Classify test signals via the internal W_hat linear classifier.

        Parameters
        ----------
        Y : (n, N_test)  CPU numpy

        Returns
        -------
        predicted_labels : (N_test,)  CPU numpy
        """
        X      = self.encode(Y)
        scores = self.W_hat @ X
        return np.argmax(scores, axis=0)

    def predict_scores(self, Y: np.ndarray) -> np.ndarray:
        """
        Return raw classifier scores (before argmax).

        Parameters
        ----------
        Y : (n, N_test)  CPU numpy

        Returns
        -------
        scores : (num_classes, N_test)  CPU numpy
        """
        X = self.encode(Y)
        return self.W_hat @ X


# ---------------------------------------------------------------------------
# Quick sanity-check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    rng = np.random.default_rng(seed=42)

    num_classes       = 3
    n_features        = 50
    n_train_per_class = 40
    n_test_per_class  = 10
    K                 = num_classes * 20
    sparsity          = 6

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

    for variant in ("lcksvd2", "lcksvd1"):
        cfg   = LCKSVDConfig(K=K, sparsity=sparsity, n_iter=10, tol=1e-6,
                             variant=variant, use_gpu=True)
        model = LCKSVD(cfg)
        model.fit(Y_train, labels_train, num_classes=num_classes)
        preds = model.predict(Y_test)
        acc   = float(np.mean(preds == labels_test))
        print(f"{variant} test accuracy: {acc * 100:.1f}%")