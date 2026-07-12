from dict_learners.dict_learner import DictLearner
from dict_learners.frozen_ksvd import FrozenKSVD


class FrozenKSVDLearner(DictLearner):
    def __init__(
            self,
            dimensions: int = 4096,
            n_frozen: int = 0,
            max_iter: int = 10,
            tol: float = 1e-6,
            n_non_zero_coefs: int = 10,
    ):
        super().__init__(name="FrozenKSVD")
        self._dictionary = None
        self.dimensions = dimensions
        self.n_frozen = n_frozen
        self.max_iter = max_iter
        self.tol = tol
        self.n_non_zero_coefs = n_non_zero_coefs
        self.frozen_ksvd = FrozenKSVD(
            n_components=self.dimensions,
            n_frozen=self.n_frozen,
            max_iter=self.max_iter,
            tol=self.tol,
            transform_n_nonzero_coefs=self.n_non_zero_coefs,
        )

    def fit(self, training_graph_embeddings, frozen_atoms=None):
        self._dictionary = self.frozen_ksvd.fit(
            training_graph_embeddings, frozen_atoms=frozen_atoms
        ).components_

        return self

    def infer(self, infer_graph_embeddings):
        sparse_embeddings = self.frozen_ksvd.transform(infer_graph_embeddings)
        return sparse_embeddings