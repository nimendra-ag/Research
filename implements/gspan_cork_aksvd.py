"""
gSpan + CORK + AKSVD Pipeline
==============================

Drop-in replacement for WL_AKSVD, using subgraph mining (gSpan)
and discriminative feature selection (CORK) instead of
Weisfeiler-Lehman hashing.

Pipeline:
    gSpan → CORK → binary indicator matrix → AKSVD → classifiers
"""

from graph_encoders.gspan_cork import GSpanCORK
from dict_learners.aksvd import AKSVD
from utils.graph_data import GraphDataLoader
from utils.evaluator import Evaluator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler
from datetime import datetime

data_loader = GraphDataLoader()

graphs, y = data_loader.nci_full_graphs, data_loader.nci_full_labels

G_train, G_test, y_train, y_test = train_test_split(
    graphs, y,
    test_size=0.2,
    random_state=42
)

G_vocab_train, G_ML_train, y_vocab_train, y_ML_train = train_test_split(
    G_train, y_train,
    test_size=0.75,
    random_state=42
)

class GSpanCORK_AKSVD:
    def __init__(self, data_loader):
        self.implementation = "GSpanCORK_AKSVD"
        self.data_loader = data_loader

    def run(
        self,
        G_vocab_train,
        y_vocab_train,
        G_ML_train,
        G_test,
        y_ML_train,
        y_test,
        # gSpan parameters
        min_support_ratio=0.10,
        max_num_vertices=10,
        min_num_vertices=2,
        # CORK parameters
        cork_tolerance=0,
        cork_max_features=None,
    ):
        start = datetime.now().strftime("%Y%m%d_%H%M%S")

        # ----- Step 1: gSpan + CORK feature extraction -----
        encoder = GSpanCORK(
            min_support_ratio=min_support_ratio,
            max_num_vertices=max_num_vertices,
            min_num_vertices=min_num_vertices,
            cork_tolerance=cork_tolerance,
            cork_max_features=cork_max_features,
            node_label_attr="feature",
            edge_label_attr=None,
            verbose=True,
        )

        # Learn the subgraph vocabulary + CORK selection on vocab split
        graph_embeddings = encoder.generate_training_embeddings(
            G_vocab_train, y_vocab_train
        )
        print(f"[Pipeline] Vocab embeddings shape: {graph_embeddings.shape}")

        # ----- Step 2: Dictionary learning (AKSVD) -----
        aksvd = AKSVD().fit(training_graph_embeddings=graph_embeddings)

        # ----- Step 3: Encode ML-train and test splits -----
        # Inference: subgraph isomorphism check against CORK-selected patterns
        graph_embeddings_ml_train = encoder.generate_inferencing_embeddings(
            G_ML_train
        )
        X_ML_train = aksvd.infer(graph_embeddings_ml_train)

        graph_embeddings_ml_test = encoder.generate_inferencing_embeddings(G_test)
        X_ML_test = aksvd.infer(graph_embeddings_ml_test)

        # ----- Step 4: Scale and evaluate -----
        scaler = MaxAbsScaler()
        X_ML_train_scaled = scaler.fit_transform(X_ML_train)
        X_ML_test_scaled = scaler.transform(X_ML_test)

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

        # ----- Save results -----
        end = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"results_gspan_cork_{start}_{end}.txt"

        # Include CORK diagnostics
        n_selected = len(encoder.get_selected_subgraphs())
        n_total = len(encoder.get_all_subgraphs())
        corr_trace = encoder.get_correspondence_trace()

        final_output = f"""
        Implementation: {self.implementation}
        gSpan: min_support_ratio={min_support_ratio}, max_vertices={max_num_vertices}
        CORK:  {n_selected} selected from {n_total} frequent subgraphs
               correspondence trace: {corr_trace}

        {results_logistic_reg}
        {results_gradient_boosting}
        {results_svm}
        {results_random_forest}
        """

        with open(f"results/{filename}", "w", encoding="utf-8") as f:
            f.write(final_output)

        print(f"Saved results to {filename}")


pipeline = GSpanCORK_AKSVD(data_loader)
pipeline.run(
    G_vocab_train, y_vocab_train,
    G_ML_train, G_test,
    y_ML_train, y_test,
    min_support_ratio=0.10,
    max_num_vertices=10,
    cork_tolerance=10,     # allow 10 unresolved correspondences
    cork_max_features=100, # safety cap
)
