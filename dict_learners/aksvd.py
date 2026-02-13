from dict_learners.dict_learner import DictLearner
from ksvd import ApproximateKSVD

class AKSVD(DictLearner):
    def __init__(
            self,
            graph_embeddings,
            dimensions: int = 128,

            max_iter: int = 10,
            tol: float = 1e-6,
            n_non_zero_coefs: int = 10,
    ):
        super().__init__(name="KSVD")
        self.aksvd = None
        self._embedding = None
        self._dictionary = None
        self.dimensions = dimensions
        self.max_iter = max_iter
        self.tol = tol
        self.graph_embeddings = graph_embeddings
        self.n_non_zero_coefs = n_non_zero_coefs

    def fit(self):
        self.aksvd = ApproximateKSVD(n_components=self.dimensions, max_iter=self.max_iter, tol=self.tol,
                 transform_n_nonzero_coefs=self.n_non_zero_coefs)
        self._dictionary = self.aksvd.fit(self.graph_embeddings).components_

        self._embedding = self.aksvd.transform(self.graph_embeddings)
        return self._embedding

