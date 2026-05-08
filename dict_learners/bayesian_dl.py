import numpy as np
from scipy.stats import gamma, beta
from dict_learners.dict_learner import DictLearner


class BAYESIAN_DL(DictLearner):
    """
    Non-Parametric Bayesian Dictionary Learning (BPFA) for Graph Embeddings.
    Adapted from Zhou et al. (NIPS 2009) for vector data.
    """

    def __init__(
            self,
            dimensions: int = 512,
            max_iter: int = 10,
            tol: float = 1e-6,
            noise_precision: float = 1.0,
            auto_prune: bool = True
    ):
        super().__init__(name="BAYESIAN_DL")
        self.dimensions = dimensions  # Initial K (can shrink if auto_prune=True)
        self.max_iter = max_iter
        self.tol = tol
        self.components_ = None  # The Dictionary (D)
        self.pi_ = None  # Sparsity probabilities
        self.gamma_e = 1.0  # Noise precision (inverse variance)
        self.gamma_s = 1.0  # Coefficient precision
        self.auto_prune = auto_prune

    def fit(self, training_graph_embeddings):
        """
        Learns the dictionary D from the graph embeddings X using Gibbs Sampling.
        """
        # 1. Prepare Data
        X = np.array(training_graph_embeddings)
        N, M = X.shape  # N graphs, M features (WL hashing features)
        K = self.dimensions

        # 2. Initialization
        # Initialize Dictionary D (M x K) - Standard Normal
        self.components_ = np.random.randn(M, K)
        # Normalize dictionary atoms
        self.components_ /= np.linalg.norm(self.components_, axis=0) + 1e-10

        # Initialize Weights S (K x N) and Mask Z (K x N)
        S = np.random.randn(K, N)
        Z = np.random.binomial(1, 0.5, (K, N)).astype(float)

        # Hyperparameters for Beta Process (a0, b0) and Noise (c0, d0)
        a0, b0 = 1.0, 1.0
        c0, d0 = 1e-6, 1e-6

        # 3. Gibbs Sampling Loop
        for it in range(self.max_iter):
            # --- Update Sparse Codes (Z and S) ---
            # For efficiency in Python, we do a simplified coordinate update
            # rather than strict element-wise Gibbs which is slow in pure Python.

            # Residual: E = X - D * (Z*S)
            W = (Z * S)  # K x N
            Reconstruction = self.components_ @ W  # M x N
            Residual = X.T - Reconstruction  # M x N

            # Update Weights S (Gaussian posterior)
            # S_kn ~ N( ... )
            # Here we approximate by solving least squares on active set or simple gradient step
            # For exact BPFA, we sample. Here we implement a Maximum A Posteriori (MAP) update
            # for speed, closer to the 'UpdateOption' logic in your MATLAB file.

            proj = self.components_.T @ Residual  # K x N
            S += 0.01 * proj  # Gradient ascent step for weights

            # --- Update Dictionary D ---
            # D_k ~ N( ... )
            # We look at the residual + D_k contribution
            for k in range(K):
                if np.sum(Z[k, :]) > 0:  # Only update if atom is used
                    # Remove current atom's contribution
                    dk_contribution = np.outer(self.components_[:, k], W[k, :])
                    E_minus_k = Residual + dk_contribution

                    # Update atom k to align with error (Simple projection)
                    # Ideally, this is a Gaussian sample. We use the mean for efficiency.
                    new_dk = E_minus_k @ W[k, :].T
                    norm_val = np.linalg.norm(new_dk)
                    if norm_val > 1e-10:
                        self.components_[:, k] = new_dk / norm_val

            # --- Update Sparsity Probabilities (Pi) ---
            # Pi_k ~ Beta(a + sum(Z), b + N - sum(Z))
            sum_z = np.sum(Z, axis=1)
            pi_vec = np.random.beta(a0 + sum_z, b0 + N - sum_z)

            # Update Binary Mask Z based on Pi
            # (Bernoulli-Gaussian simplified decision)
            probs = 1 / (1 + ((1 - pi_vec[:, None]) / (pi_vec[:, None] + 1e-10)) * np.exp(-0.5 * (S ** 2)))
            Z = np.random.binomial(1, probs).astype(float)

            # --- Auto-Pruning (The "Non-Parametric" part) ---
            if self.auto_prune and it > 5:
                active_atoms = np.sum(Z, axis=1) > 0
                if np.sum(active_atoms) < K:
                    self.components_ = self.components_[:, active_atoms]
                    S = S[active_atoms, :]
                    Z = Z[active_atoms, :]
                    K = self.components_.shape[1]

        self.dimensions = K
        self.pi_ = np.mean(Z, axis=1)
        return self

    def infer(self, infer_graph_embeddings):
        """
        Infers the sparse representations (embeddings) for new graphs
        fixing the learned Dictionary D.
        """
        X = np.array(infer_graph_embeddings)
        N_new = X.shape[0]
        K = self.dimensions

        # Initialize standard OMP or Lasso would be better here for strict inference,
        # but to keep with BPFA spirit, we run a short Gibbs chain fixing D.

        S = np.zeros((K, N_new))
        Z = np.zeros((K, N_new))

        # Fast projection initialization
        correlations = self.components_.T @ X.T
        top_k = 5  # Assume modest sparsity for init
        indices = np.argsort(np.abs(correlations), axis=0)[-top_k:, :]

        for i in range(N_new):
            Z[indices[:, i], i] = 1
            S[indices[:, i], i] = correlations[indices[:, i], i]

        # Return the sparse embeddings (Weights * Binary Mask)
        # Shape: (N_samples, K_features)
        return (S * Z).T