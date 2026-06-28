"""
utils/lcksvd_evaluator.py
=========================
Evaluates LC-KSVD's built-in linear classifier W_hat on held-out test data.

This is separate from the pipeline's Evaluator class, which trains and
evaluates external ML models (LR, GB, SVM, RF) on top of the sparse codes.
Here we instead use the classifier that LC-KSVD learned jointly with its
dictionary — W_hat — and report a full set of classification metrics.

Why both evaluators matter
--------------------------
The external Evaluator (LR, GB, etc.) tells you how good the *sparse codes*
are as features for arbitrary classifiers.
This evaluator tells you how good LC-KSVD's *own* classifier is — i.e. how
well the joint learning of D, A, and W actually worked end-to-end.
Comparing the two gives insight into whether the internal W_hat is the
bottleneck or whether the sparse codes themselves need improvement.

Metrics reported
----------------
For binary classification (the NCI dataset is binary):
  - Precision    : TP / (TP + FP)  — of all positive predictions, how many correct
  - Recall       : TP / (TP + FN)  — of all actual positives, how many caught
  - F1-Score     : harmonic mean of Precision and Recall
  - ROC-AUC      : area under the ROC curve (uses raw classifier scores)
  - PR-AUC       : area under the Precision-Recall curve (better for imbalance)

For imbalanced datasets PR-AUC is more informative than ROC-AUC because
ROC-AUC can look deceptively high when the majority class dominates.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    classification_report,
)
from sklearn.preprocessing import label_binarize


class LCKSVDEvaluator:
    """
    Evaluates the LC-KSVD internal linear classifier W_hat.

    Parameters
    ----------
    lcksvd_learner : LCKSVDLearner
        A fitted LCKSVDLearner instance (from dict_learners/lcksvd.py).
        The internal model is accessed via lcksvd_learner.lcksvd_model.
    """

    def __init__(self, lcksvd_learner) -> None:
        self._model = lcksvd_learner.lcksvd_model

        if self._model.D_hat is None:
            raise RuntimeError(
                "LCKSVDLearner is not fitted yet. Call .fit() before evaluating."
            )

        self._num_classes = self._model.num_classes_

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_predictions(
        self,
        graph_embeddings: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Run the LC-KSVD internal classifier on WL graph embeddings.

        Steps:
          1. Transpose (N, n) → (n, N) for the core model convention.
          2. encode()         → sparse codes X  (K, N)
          3. W_hat @ X        → raw scores      (num_classes, N)
          4. argmax           → predicted labels (N,)

        Parameters
        ----------
        graph_embeddings : (N, n) ndarray from WL.generate_inferencing_embeddings()

        Returns
        -------
        y_pred   : (N,) integer predicted class labels
        y_scores : (N, num_classes) raw classifier scores (before argmax)
                   Used for ROC-AUC and PR-AUC computation.
        """
        # Transpose to column-major for the core LCKSVD convention
        Y = np.array(graph_embeddings, dtype=float).T          # (n, N)

        # Sparse codes via OMP on the learned dictionary D_hat
        X = self._model.encode(Y)                              # (K, N)

        # Raw scores from the linear classifier W_hat
        raw_scores = self._model.W_hat @ X                    # (num_classes, N)

        y_pred = np.argmax(raw_scores, axis=0)                # (N,)
        y_scores = raw_scores.T                               # (N, num_classes)

        return y_pred, y_scores

    # ------------------------------------------------------------------
    # Public evaluation method
    # ------------------------------------------------------------------

    def evaluate(
        self,
        graph_embeddings_test: np.ndarray,
        y_test: np.ndarray,
        positive_label: int = 1,
    ) -> dict:
        """
        Compute Precision, Recall, F1, ROC-AUC, and PR-AUC for the
        LC-KSVD internal classifier on held-out test graphs.

        Parameters
        ----------
        graph_embeddings_test : (N_test, n) ndarray
            WL embeddings for test graphs, from wl.generate_inferencing_embeddings().
            These must NOT be scaled — the LC-KSVD classifier operates
            directly in the sparse code space, not the MaxAbsScaler space
            used by the external Evaluator.

        y_test : (N_test,) array-like
            True class labels.

        positive_label : int
            Which class index is considered "positive" for binary metrics.
            Defaults to 1. For the NCI dataset the minority class is 1.

        Returns
        -------
        metrics : dict with keys:
            'precision', 'recall', 'f1', 'roc_auc', 'pr_auc',
            'y_pred', 'y_scores', 'classification_report'
        """
        y_test = np.array(y_test, dtype=int)
        y_pred, y_scores = self._get_predictions(graph_embeddings_test)

        is_binary = (self._num_classes == 2)

        if is_binary:
            # For binary classification, use scores of the positive class
            # for threshold-based metrics (ROC-AUC, PR-AUC).
            # Softmax is not applied — raw W_hat scores are monotone enough
            # for ranking purposes, which is all AUC metrics require.
            pos_scores = y_scores[:, positive_label]

            precision = precision_score(
                y_test, y_pred,
                pos_label=positive_label,
                zero_division=0,
            )
            recall = recall_score(
                y_test, y_pred,
                pos_label=positive_label,
                zero_division=0,
            )
            f1 = f1_score(
                y_test, y_pred,
                pos_label=positive_label,
                zero_division=0,
            )
            roc_auc = roc_auc_score(y_test, pos_scores)

            # average_precision_score computes PR-AUC directly
            pr_auc = average_precision_score(y_test, pos_scores, pos_label=positive_label)

        else:
            # Multiclass: use macro averaging for Precision, Recall, F1
            # and OvR (one-vs-rest) for AUC metrics.
            precision = precision_score(
                y_test, y_pred, average="macro", zero_division=0
            )
            recall = recall_score(
                y_test, y_pred, average="macro", zero_division=0
            )
            f1 = f1_score(
                y_test, y_pred, average="macro", zero_division=0
            )

            # Binarise labels for multiclass AUC (OvR)
            classes = np.arange(self._num_classes)
            y_bin = label_binarize(y_test, classes=classes)  # (N, num_classes)

            roc_auc = roc_auc_score(y_bin, y_scores, multi_class="ovr", average="macro")
            pr_auc = average_precision_score(y_bin, y_scores, average="macro")

        report = classification_report(
            y_test, y_pred, zero_division=0
        )

        metrics = {
            "precision": precision,
            "recall":    recall,
            "f1":        f1,
            "roc_auc":   roc_auc,
            "pr_auc":    pr_auc,
            "y_pred":    y_pred,
            "y_scores":  y_scores,
            "classification_report": report,
        }

        return metrics

    def print_results(self, metrics: dict) -> None:
        """
        Print a formatted summary of the evaluation metrics.

        Parameters
        ----------
        metrics : dict returned by .evaluate()
        """
        print("=" * 52)
        print("   LC-KSVD Internal Classifier — Evaluation")
        print("=" * 52)
        print(f"  Precision  : {metrics['precision']:.4f}")
        print(f"  Recall     : {metrics['recall']:.4f}")
        print(f"  F1-Score   : {metrics['f1']:.4f}")
        print(f"  ROC-AUC    : {metrics['roc_auc']:.4f}")
        print(f"  PR-AUC     : {metrics['pr_auc']:.4f}")
        print("-" * 52)
        print("  Per-class breakdown:")
        print(metrics["classification_report"])
        print("=" * 52)
