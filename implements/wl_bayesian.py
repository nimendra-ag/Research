"""
WL + BPFA pipeline for graph classification.

Uses Weisfeiler-Leman graph encoding followed by Beta Process Factor Analysis
(Bayesian dictionary learning) for sparse graph representations, then evaluates
multiple ML classifiers.
"""

from dict_learners.bayesian_dl import BPFA
from graph_encoders.wl import WL
from utils.graph_data import GraphDataLoader
from utils.evaluator import Evaluator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler
from datetime import datetime
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

data_loader = GraphDataLoader()

graphs, y = data_loader.nci_full_graphs, data_loader.nci_full_labels

G_train, G_test, y_train, y_test = train_test_split(
    graphs, y,
    test_size=0.2,
    random_state=42,
)

G_vocab_train, G_ML_train, y_vocab_train, y_ML_train = train_test_split(
    G_train, y_train,
    test_size=0.75,
    random_state=42,
)


class WL_BPFA:
    def __init__(self, data_loader):
        self.implementation = "WL_BPFA"
        self.data_loader = data_loader

    def run(
        self,
        G_vocab_train,
        y_vocab_train,
        G_ML_train,
        G_test,
        y_ML_train,
        y_test,
    ):
        start = datetime.now().strftime("%Y%m%d_%H%M%S")

        # --- Graph encoding ---
        wl = WL()
        graph_embeddings = wl.generate_training_embeddings(
            G_vocab_train, y_vocab_train
        )

        # --- Bayesian dictionary learning ---
        bpfa = BPFA(
            n_components=64,
            n_iter=10,
            n_infer_iter=10,
            n_burnin=20,
            n_collect=10,
            random_state=42,
            verbose=True,
        ).fit(training_graph_embeddings=graph_embeddings)

        logger.info(
            "Effective dictionary size: %d / %d",
            bpfa.effective_dictionary_size,
            bpfa.n_components,
        )
        logger.info("Estimated noise std: %.4f", bpfa.noise_std)

        # --- Sparse codes for ML training set ---
        graph_embeddings_ml_train = wl.generate_inferencing_embeddings(G_ML_train)
        X_ML_train = bpfa.infer(graph_embeddings_ml_train)

        # --- Sparse codes for ML test set ---
        graph_embeddings_ml_test = wl.generate_inferencing_embeddings(G_test)
        X_ML_test = bpfa.infer(graph_embeddings_ml_test)

        # --- Scaling ---
        scaler = MaxAbsScaler()
        X_ML_train_scaled = scaler.fit_transform(X_ML_train)
        X_ML_test_scaled = scaler.transform(X_ML_test)

        # --- Model evaluation ---
        evaluator = Evaluator(
            X_ML_train_scaled, y_ML_train, X_ML_test_scaled, y_test
        )

        results_logistic_reg = evaluator.predict_logistic_regression()
        print(results_logistic_reg)

        results_gradient_boosting = evaluator.predict_gradient_boosting()
        print(results_gradient_boosting)

        results_svm = evaluator.predict_svm()
        print(results_svm)

        results_random_forest = evaluator.predict_random_forest()
        print(results_random_forest)

        final_output = f"""
            {results_logistic_reg}
            {results_gradient_boosting}
            {results_svm}
            {results_random_forest}
            """

        end = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"results_wlbayesian_{start}_{end}.txt"

        os.makedirs("results", exist_ok=True)
        with open(f"results/{filename}", "w", encoding="utf-8") as f:
            f.write(final_output)

        print(f"Saved results to {filename}")


wl_bpfa = WL_BPFA(data_loader)
wl_bpfa.run(
    G_vocab_train, y_vocab_train, G_ML_train, G_test, y_ML_train, y_test
)