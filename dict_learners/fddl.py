import numpy as np
import scipy.optimize as opt
from dict_learners.dict_learner import DictLearner


class FDDL(DictLearner):
    def __init__(
            self,
            k: int = 32,
            lambda1: float = 0.1,
            lambda2: float = 0.1,
            eta: float = 1.0,
            max_iter: int = 64,
            lr: float = 0.01,
            ipm_iters: int = 15
    ):
        super().__init__(name="FDDL")
        self.k = k  # Atoms per class sub-dictionary
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.eta = eta
        self.max_iter = max_iter
        self.lr = lr
        self.ipm_iters = ipm_iters

        self.D = None
        self.X_train = None
        self.classes_ = None
        self.class_sizes_ = None
        self.M_i = {}  # Store mean vectors for native FDDL classification

    def _soft_threshold(self, X: np.ndarray, tau: float) -> np.ndarray:
        return np.sign(X) * np.maximum(np.abs(X) - tau, 0.0)

    def _compute_gradient_Xi(self, Ai, D, Xi, class_idx, k, lambda2, eta, M_global):
        grad_global = -2 * D.T @ (Ai - D @ Xi)

        grad_local = np.zeros_like(Xi)
        start_idx = class_idx * k
        end_idx = start_idx + k
        Di = D[:, start_idx:end_idx]
        Xii = Xi[start_idx:end_idx, :]
        grad_local[start_idx:end_idx, :] = -2 * Di.T @ (Ai - Di @ Xii)

        grad_sabotage = np.zeros_like(Xi)
        for j in range(D.shape[1] // k):
            if j != class_idx:
                j_start = j * k
                j_end = j_start + k
                Dj = D[:, j_start:j_end]
                Xij = Xi[j_start:j_end, :]
                grad_sabotage[j_start:j_end, :] = 2 * Dj.T @ (Dj @ Xij)

        Mi = np.mean(Xi, axis=1, keepdims=True)
        grad_fisher = 2 * (Xi - Mi) - 2 * (Mi - M_global) + 2 * eta * Xi

        return grad_global + grad_local + grad_sabotage + (lambda2 * grad_fisher)

    def _update_X(self, A, D, X, k, n_classes, class_sizes):
        M_global = np.mean(X, axis=1, keepdims=True)
        col_start = 0
        for i in range(n_classes):
            col_end = col_start + class_sizes[i]
            Ai = A[:, col_start:col_end]
            Xi = X[:, col_start:col_end]

            for _ in range(self.ipm_iters):
                grad = self._compute_gradient_Xi(Ai, D, Xi, i, k, self.lambda2, self.eta, M_global)
                Xi = Xi - self.lr * grad
                Xi = self._soft_threshold(Xi, self.lambda1 * self.lr)

            X[:, col_start:col_end] = Xi
            col_start = col_end
        return X

    def _update_D(self, A, D, X, k, n_classes, class_sizes):
        for i in range(n_classes):
            start_idx = i * k
            end_idx = start_idx + k
            Di = D[:, start_idx:end_idx]
            Xi_all = X[start_idx:end_idx, :]

            A_hat = A.copy()
            for j in range(n_classes):
                if j != i:
                    j_start = j * k
                    j_end = j_start + k
                    A_hat -= D[:, j_start:j_end] @ X[j_start:j_end, :]

            col_start = sum(class_sizes[:i])
            col_end = col_start + class_sizes[i]
            Ai = A[:, col_start:col_end]
            Xii = Xi_all[:, col_start:col_end]

            X_others = np.delete(Xi_all, np.s_[col_start:col_end], axis=1)
            zeros = np.zeros((A.shape[0], X_others.shape[1]))

            Lambda_i = np.hstack((A_hat, Ai, zeros))
            Zi = np.hstack((Xi_all, Xii, X_others))

            for atom_idx in range(k):
                d_l = Di[:, atom_idx].reshape(-1, 1)
                z_l = Zi[atom_idx, :].reshape(1, -1)

                Y = Lambda_i - (Di @ Zi) + (d_l @ z_l)
                d_new = Y @ z_l.T
                norm_d = np.linalg.norm(d_new)
                Di[:, atom_idx] = (d_new / norm_d).flatten() if norm_d > 1e-10 else d_l.flatten()

            D[:, start_idx:end_idx] = Di
        return D

    def fit(self, training_graph_embeddings, y_train):
        print(f"Training {self.name}...")
        # Group data by class for FDDL contiguous block assumption
        self.classes_ = np.unique(y_train)
        n_classes = len(self.classes_)

        A_grouped = []
        self.class_sizes_ = []
        for c in self.classes_:
            A_c = training_graph_embeddings[y_train == c].T
            A_grouped.append(A_c)
            self.class_sizes_.append(A_c.shape[1])

        A = np.hstack(A_grouped)
        features = A.shape[0]
        total_atoms = self.k * n_classes

        # Initialize D
        self.D = np.zeros((features, total_atoms))
        col_start = 0
        for i in range(n_classes):
            Ai = A[:, col_start:col_start + self.class_sizes_[i]]
            idx = np.random.choice(self.class_sizes_[i], self.k, replace=True)
            Di = Ai[:, idx]
            Di = Di / np.linalg.norm(Di, axis=0)
            self.D[:, i * self.k:(i + 1) * self.k] = Di
            col_start += self.class_sizes_[i]

        X = np.zeros((total_atoms, sum(self.class_sizes_)))

        # Alternating Optimization Loop
        for it in range(self.max_iter):
            X = self._update_X(A, self.D, X, self.k, n_classes, self.class_sizes_)
            self.D = self._update_D(A, self.D, X, self.k, n_classes, self.class_sizes_)

        self.X_train = X

        # Store mean coefficients per class for native FDDL prediction
        col_start = 0
        for i, c in enumerate(self.classes_):
            col_end = col_start + self.class_sizes_[i]
            Xi = X[:, col_start:col_end]
            self.M_i[c] = np.mean(Xi, axis=1)
            col_start = col_end

        return self

    def infer(self, infer_graph_embeddings):
        """Uses ISTA to find sparse codes for new testing data over the global learned D"""
        A_test = infer_graph_embeddings.T
        Z = np.zeros((self.D.shape[1], A_test.shape[1]))

        # Standard sparse coding inference using learned Dictionary
        for _ in range(self.ipm_iters * 2):  # Run a bit longer for inference
            grad = -2 * self.D.T @ (A_test - self.D @ Z)
            Z = Z - self.lr * grad
            Z = self._soft_threshold(Z, self.lambda1 * self.lr)

        return Z.T  # Return shape (n_samples, total_atoms)