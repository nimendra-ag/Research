from dict_learners.dict_learner import DictLearner
from dict_learners.incremental_frozen_dict import IncrementalFrozenDictionary
import numpy as np


class FrozenKSVDLearner(DictLearner):
    def __init__(
            self,
            n_components_base: int = 96,
            n_components_residual: int = 32,
            max_iter: int = 10,
            tol: float = 1e-6,
            n_non_zero_coefs: int = 10,
            base_label: int = -1,
    ):
        """
        Parameters
        ----------
        n_components_base:
            Number of atoms for the base (majority) class dictionary.

        n_components_residual:
            Number of new atoms to learn for each additional class.

        max_iter:
            Maximum K-SVD iterations per stage.

        tol:
            Convergence tolerance.

        n_non_zero_coefs:
            Sparsity level for OMP.

        base_label:
            The class label to use as the base dictionary. Defaults to -1
            to match the WL encoder's majority class convention.  All
            other labels are added incrementally via add_class.
        """
        super().__init__(name="FrozenKSVD")
        self._dictionary = None
        self.n_components_base = n_components_base
        self.n_components_residual = n_components_residual
        self.max_iter = max_iter
        self.tol = tol
        self.n_non_zero_coefs = n_non_zero_coefs
        self.base_label = base_label
        self.incremental = IncrementalFrozenDictionary(
            n_components_base=self.n_components_base,
            n_components_residual=self.n_components_residual,
            max_iter=self.max_iter,
            tol=self.tol,
            transform_n_nonzero_coefs=self.n_non_zero_coefs,
        )

    def fit(self, training_graph_embeddings, labels):
        """
        Build the dictionary incrementally: base class first, then
        each remaining class adds residual atoms on top.

        Parameters
        ----------
        training_graph_embeddings: shape = [n_samples, n_features]
        labels: shape = [n_samples]
            Class labels used to split data by class.

        Returns
        -------
        self
        """
        labels = np.asarray(labels)
        unique_labels = np.unique(labels)

        # Determine base label — use configured base_label if present,
        # otherwise fall back to the first unique label
        if self.base_label in unique_labels:
            base_label = self.base_label
        else:
            base_label = unique_labels[0]

        # Stage 1: learn base dictionary from base class
        base_mask = labels == base_label
        base_data = training_graph_embeddings[base_mask]
        self.incremental.fit_base(base_data)

        # Stage 2+: add residual atoms for each remaining class
        for cls in unique_labels:
            if cls == base_label:
                continue
            cls_mask = labels == cls
            cls_data = training_graph_embeddings[cls_mask]
            self.incremental.add_class(cls_data)

        self._dictionary = self.incremental.components_
        return self

    def infer(self, infer_graph_embeddings):
        sparse_embeddings = self.incremental.transform(infer_graph_embeddings)
        return sparse_embeddings
