"""
dict_learners/lcksvd.py
=======================
Pipeline adapter that wraps the GPU-accelerated LCKSVD implementation
into the DictLearner interface used by the WL pipeline.

The adapter is identical to the CPU version — only the import source
changes (lc_ksvd_gpu instead of lc_ksvd).

All GPU <-> CPU transfers are handled inside lc_ksvd_gpu.py.
fit(), infer(), and the rest of the pipeline always receive plain
numpy arrays from this class, so nothing else in the pipeline changes.

To switch back to CPU-only: change the import to:
    from dict_learners.lc_ksvd import LCKSVD, LCKSVDConfig
"""

import numpy as np
from dict_learners.dict_learner import DictLearner
from dict_learners.lc_ksvd_gpu import LCKSVD, LCKSVDConfig   # PyTorch GPU version   # GPU-accelerated version


class LCKSVDLearner(DictLearner):
    """
    Supervised dictionary learner wrapping LCKSVD for the WL pipeline.

    Parameters
    ----------
    K        : int   — number of dictionary atoms
    sparsity : int   — max nonzero coefficients per sparse code (T)
    n_iter   : int   — outer LC-KSVD iterations
    alpha    : float — weight of discriminative sparse-code error term
    beta     : float — weight of classification error term
    lambda1  : float — ridge regularisation for classifier W
    lambda2  : float — ridge regularisation for transform A
    variant  : str   — 'lcksvd1' or 'lcksvd2'
    use_gpu  : bool  — True to use CUDA GPU via CuPy (falls back to CPU
                       automatically if CuPy is not installed or no GPU found)
    """

    def __init__(
        self,
        K: int = 128,
        sparsity: int = 10,
        n_iter: int = 10,
        alpha: float = 16.0,
        beta: float = 4.0,
        lambda1: float = 1e-4,
        lambda2: float = 1e-4,
        variant: str = "lcksvd2",
        use_gpu: bool = True,
    ):
        super().__init__(name="LCKSVD")

        self._config = LCKSVDConfig(
            K=K,
            sparsity=sparsity,
            n_iter=n_iter,
            alpha=alpha,
            beta=beta,
            lambda1=lambda1,
            lambda2=lambda2,
            variant=variant,
            use_gpu=use_gpu,
        )

        self.lcksvd_model: LCKSVD = LCKSVD(self._config)
        self._dictionary = None

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(
        self,
        training_graph_embeddings: np.ndarray,
        labels: np.ndarray,
    ) -> "LCKSVDLearner":
        """
        Learn the dictionary from labelled WL graph embeddings.

        Parameters
        ----------
        training_graph_embeddings : (N, n) ndarray — CPU numpy array from WL
        labels                    : (N,) integer class indices

        Returns
        -------
        self
        """
        # WL produces (N, n); LCKSVD core expects (n, N)
        Y = np.array(training_graph_embeddings, dtype=float).T   # (n, N)

        labels_arr = np.array(labels, dtype=int)
        num_classes = int(np.unique(labels_arr).shape[0])

        # GPU transfer happens inside lcksvd_model.fit()
        self.lcksvd_model.fit(Y, labels_arr, num_classes=num_classes)

        self._dictionary = self.lcksvd_model.D_hat   # (n, K) — CPU numpy
        return self

    # ------------------------------------------------------------------
    # infer
    # ------------------------------------------------------------------

    def infer(self, infer_graph_embeddings: np.ndarray) -> np.ndarray:
        """
        Encode graphs into sparse codes using the learned dictionary.

        Parameters
        ----------
        infer_graph_embeddings : (N, n) ndarray — CPU numpy from WL

        Returns
        -------
        sparse_embeddings : (N, K) ndarray — CPU numpy, ready for sklearn
        """
        Y = np.array(infer_graph_embeddings, dtype=float).T   # (n, N)

        # encode() moves Y to GPU internally and returns CPU numpy
        X = self.lcksvd_model.encode(Y)   # (K, N) CPU numpy
        return X.T                         # (N, K) CPU numpy
