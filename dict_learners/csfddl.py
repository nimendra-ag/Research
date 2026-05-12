import numpy as np
import scipy.optimize as opt
from dict_learners.dict_learner import DictLearner

class CSFDDL(DictLearner):
    def __init__(
            self,
            k: int = 10,
            lambda1: float = 0.05,
            lambda2: float = 0.01,
            eta: float = 1.0,
            max_iter: int = 20,
            lr: float = 0.01,
            ipm_iters: int = 15,
            cost_sensitive: bool = True # NEW: Toggle for Cost-Sensitive FDDL
    ):
        super().__init__(name="CS-FDDL" if cost_sensitive else "FDDL")
        self.k = k  # Atoms per class sub-dictionary
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.eta = eta
        self.max_iter = max_iter
        self.lr = lr
        self.ipm_iters = ipm_iters
        self.cost_sensitive = cost_sensitive
        self.class_weights_ = None
        
        self.D = None
        self.X_train = None
        self.classes_ = None
        self.class_sizes_ = None
        self.M_i = {} # Store mean vectors for native FDDL classification

    def _soft_threshold(self, X: np.ndarray, tau: float) -> np.ndarray:
        return np.sign(X) * np.maximum(np.abs(X) - tau, 0.0)

    def _compute_gradient_Xi(self, Ai, D, Xi, class_idx, k, lambda2, eta, M_global_w, w_i):
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
        grad_fisher = 2 * (Xi - Mi) - 2 * (Mi - M_global_w) + 2 * eta * Xi
    
        # Multiply the entire gradient by the cost-sensitive weight
        total_grad = grad_global + grad_local + grad_sabotage + (lambda2 * grad_fisher)
        return w_i * total_grad

    def _update_X(self, A, D, X, k, n_classes, class_sizes):
        # Calculate Weighted Global Mean (m_w)
        class_means = []
        col_start = 0
        for i in range(n_classes):
            col_end = col_start + class_sizes[i]
            class_means.append(np.mean(X[:, col_start:col_end], axis=1, keepdims=True))
            col_start = col_end
        M_global_w = np.mean(class_means, axis=0) # Mean of the class means

        col_start = 0
        for i in range(n_classes):
            col_end = col_start + class_sizes[i]
            Ai = A[:, col_start:col_end]
            Xi = X[:, col_start:col_end]
            w_i = self.class_weights_[i]

            # Adjust learning rate to prevent exploding gradients on minority classes
            effective_lr = self.lr / np.sqrt(w_i) if self.cost_sensitive else self.lr

            for _ in range(self.ipm_iters):
                grad = self._compute_gradient_Xi(Ai, D, Xi, i, k, self.lambda2, self.eta, M_global_w, w_i)
                Xi = Xi - effective_lr * grad
                Xi = self._soft_threshold(Xi, self.lambda1 * effective_lr)

            X[:, col_start:col_end] = Xi
            col_start = col_end
        return X

    def _update_D(self, A, D, X, k, n_classes, class_sizes):
        # Full scaling vector for all samples in A
        full_scaling_vec = np.repeat(np.sqrt(self.class_weights_), class_sizes)

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

            # Construct Sabotage scaling vector
            weights_others = np.delete(self.class_weights_, i)
            sizes_others = np.delete(class_sizes, i)
            sabotage_scaling_vec = np.repeat(np.sqrt(weights_others), sizes_others)

            Lambda_i = np.hstack((A_hat, Ai, zeros))
            Zi = np.hstack((Xi_all, Xii, X_others))

            # Apply Cost-Sensitive Scaling
            scale_array = np.concatenate([
                full_scaling_vec,                                           # Scales A_hat and Xi_all
                np.repeat(np.sqrt(self.class_weights_[i]), class_sizes[i]), # Scales Ai and Xii
                sabotage_scaling_vec                                        # Scales zeros and X_others
            ])
            
            Lambda_i_scaled = Lambda_i * scale_array
            Zi_scaled = Zi * scale_array

            for atom_idx in range(k):
                d_l = Di[:, atom_idx].reshape(-1, 1)
                z_l = Zi_scaled[atom_idx, :].reshape(1, -1) # Use scaled Z

                # Use scaled Lambda and Z
                Y = Lambda_i_scaled - (Di @ Zi_scaled) + (d_l @ z_l)
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
        
        total_samples = sum(self.class_sizes_)
        
        if self.cost_sensitive:
            self.class_weights_ = np.array([total_samples / (n_classes * size) for size in self.class_sizes_])
        else:
            self.class_weights_ = np.ones(n_classes)
            
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
        for _ in range(self.ipm_iters * 2): # Run a bit longer for inference
            grad = -2 * self.D.T @ (A_test - self.D @ Z)
            Z = Z - self.lr * grad
            Z = self._soft_threshold(Z, self.lambda1 * self.lr)
            
        return Z.T # Return shape (n_samples, total_atoms)