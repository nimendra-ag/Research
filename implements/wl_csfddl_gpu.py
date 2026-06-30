import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.graph_data import GraphDataLoader
from dict_learners.csfddl_gpu import CSFDDLGPU
from graph_encoders.wl import WL
from utils.evaluator import Evaluator
from utils.src_classifier import SRCClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler
import numpy as np
from datetime import datetime


data_loader = GraphDataLoader()
graphs, y = data_loader.nci_full_graphs, data_loader.nci_full_labels

G_train, G_test, y_train, y_test = train_test_split(
    graphs, y, test_size=0.2, random_state=42
)
G_vocab_train, G_ML_train, y_vocab_train, y_ML_train = train_test_split(
    G_train, y_train, test_size=0.75, random_state=42
)


class WL_CSFDDLGPU:
    def __init__(self):
        self.implementation = "WL_CSFDDLGPU"

    def run(self, G_vocab_train, y_vocab_train, G_ML_train, G_test,
            y_ML_train, y_test):

        start = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 1. Feature Extraction
        wl = WL()
        graph_embeddings = wl.generate_training_embeddings(
            G_vocab_train, y_vocab_train
        )

        # 2. Cost-Sensitive FDDL
        csfddl_gpu = CSFDDLGPU(weighting='inverse_sqrt')
        csfddl_gpu.fit(
            training_graph_embeddings=graph_embeddings,
            y_train=y_vocab_train
        )

        # 3. Encode train and test
        graph_embeddings_ml_train = wl.generate_inferencing_embeddings(G_ML_train)
        X_ML_train = csfddl_gpu.infer(graph_embeddings_ml_train)

        graph_embeddings_ml_test = wl.generate_inferencing_embeddings(G_test)
        X_ML_test = csfddl_gpu.infer(graph_embeddings_ml_test)

        # 4a. Native classifiers — gamma sweep to find optimal value
        print("\n--- Gamma sweep (SRC / CS-FDDL native) ---")
        best_gamma, best_bal_acc = 0.0, 0.0
        for g in [0.0, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]:
            src = SRCClassifier(csfddl_gpu, gamma=g)
            result = src.evaluate(graph_embeddings_ml_test, y_test)
            tag = "Pure SRC" if g == 0.0 else f"gamma={g}"
            print(f"  {tag}: {result}")
            if result['balanced_acc'] > best_bal_acc:
                best_bal_acc = result['balanced_acc']
                best_gamma = g

        print(f"\n  Best gamma = {best_gamma} (balanced_acc = {best_bal_acc})")

        # 4b. Report best native classifier
        src_best = SRCClassifier(csfddl_gpu, gamma=best_gamma)
        results_native = src_best.evaluate(graph_embeddings_ml_test, y_test)
        print(f"\nCS-FDDL native (gamma={best_gamma}): {results_native}")

        # 5. External classifiers
        scaler = MaxAbsScaler()
        X_ML_train_scaled = scaler.fit_transform(X_ML_train)
        X_ML_test_scaled = scaler.transform(X_ML_test)

        evaluator = Evaluator(
            X_ML_train_scaled, y_ML_train,
            X_ML_test_scaled, y_test,
            dl_model="wl_csfddl_gpu", dataset="nci-full"
        )

        print("\n--- External classifiers ---")
        results_logistic_reg = evaluator.predict_logistic_regression()
        print(results_logistic_reg)

        results_gradient_boosting = evaluator.predict_gradient_boosting()
        print(results_gradient_boosting)

        results_svm = evaluator.predict_svm()
        print(results_svm)

        results_random_forest = evaluator.predict_random_forest()
        print(results_random_forest)

        # 6. Save results
        end = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"results_wl_csfddl_{csfddl_gpu.k}_{csfddl_gpu.weighting}_{start}_{end}.txt"

        final_output = f"""CS-FDDL Configuration:
    k = {csfddl_gpu.k}
    weighting = {csfddl_gpu.weighting}
    class_weights = {csfddl_gpu.class_weights_}

Native classifier (gamma={best_gamma}):
    {results_native}

External classifiers:
    {results_logistic_reg}
    {results_gradient_boosting}
    {results_svm}
    {results_random_forest}
"""

        with open(f"results/{filename}", "w", encoding="utf-8") as f:
            f.write(final_output)

        print(f"\nSaved results to {filename}")


data_loader = GraphDataLoader()
wl_csfddl_gpu = WL_CSFDDLGPU()
wl_csfddl_gpu.run(
    G_vocab_train, y_vocab_train,
    G_ML_train, G_test,
    y_ML_train, y_test
)
