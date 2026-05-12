import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dict_learners.csfddl import CSFDDL
from graph_encoders.wl import WL
from utils.graph_data import GraphDataLoader
from utils.evaluator import Evaluator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler
import numpy as np

# Configuration
# DATASET = "nci-full"
# MODEL_NAME = "csfddl"

# data_loader = GraphDataLoader()

# # Load graph data
# graphs, y = data_loader.nci_full_graphs, data_loader.nci_full_labels
# y = np.array(y)

# # First divide the data into train and test sets.
# G_train, G_test, y_train, y_test = train_test_split(graphs, y, test_size=0.2, random_state=42)

# # Divide the train set further into vocab training and ML training sets
# G_vocab_train, G_ML_train, y_vocab_train, y_ML_train = train_test_split(G_train, y_train, test_size=0.75, random_state=42)

# # 1. Feature Extraction (WL)
# wl = WL()
# graph_embeddings = wl.generate_training_embeddings(G_vocab_train)
# graph_embeddings_ml_train = wl.generate_inferencing_embeddings(G_ML_train)
# graph_embeddings_ml_test = wl.generate_inferencing_embeddings(G_test)

# # 2. Dictionary Learning (CSFDDL)
# csfddl = CSFDDL(k=10, max_iter=20)
# # Note: CSFDDL is supervised, so it needs labels
# csfddl.fit(training_graph_embeddings=graph_embeddings_ml_train, y_train=y_ML_train)

# # 3. Infer Sparse Coefficients
# X_ML_train = csfddl.infer(graph_embeddings_ml_train)
# X_ML_test = csfddl.infer(graph_embeddings_ml_test)

# # Scale
# scaler = MaxAbsScaler()
# X_ML_train_scaled = scaler.fit_transform(X_ML_train)
# X_ML_test_scaled = scaler.transform(X_ML_test)

# # 4. Model Evaluation
# evaluator = Evaluator(
#     X_train=X_ML_train_scaled, 
#     y_train=y_ML_train, 
#     X_test=X_ML_test_scaled, 
#     y_test=y_test, 
#     dl_model=MODEL_NAME, 
#     dataset=DATASET
# )

# results_logistic_reg = evaluator.predict_logistic_regression()
# print("Logistic Regression:", results_logistic_reg)

# results_gradient_boosting = evaluator.predict_gradient_boosting()
# print("Gradient Boosting:", results_gradient_boosting)



class WL_CSFDDL:
    def __init__(self, data_loader):
        self.implementation = "WL_CSFDDL"
        self.data_loader = data_loader

    def run(self):

        # load graph data
        graphs, y = self.data_loader.nci_full_graphs, self.data_loader.nci_full_labels
        # First divide the data into train and test sets.
        G_train, G_test, y_train, y_test = train_test_split(graphs, y, test_size=0.2, random_state=42)

        # divide the train set further into vocab training and ML training sets
        G_vocab_train, G_ML_train, y_vocab_train, y_ML_train = train_test_split(G_train, y_train, test_size=0.75,
                                                                                random_state=42)

        wl = WL()
        graph_embeddings = wl.generate_training_embeddings(G_vocab_train)

        csfddl = CSFDDL(k=10, max_iter=50, lambda1=0.01, lambda2=0.1, cost_sensitive=True)
        csfddl.fit(training_graph_embeddings=graph_embeddings, y_train=y_vocab_train)

        graph_embeddings_ml_train = wl.generate_inferencing_embeddings(G_ML_train)
        X_ML_train = csfddl.infer(graph_embeddings_ml_train)

        graph_embeddings_ml_test = wl.generate_inferencing_embeddings(G_test)
        X_ML_test = csfddl.infer(graph_embeddings_ml_test)

        scaler = MaxAbsScaler()
        X_ML_train_scaled = scaler.fit_transform(X_ML_train)
        X_ML_test_scaled = scaler.transform(X_ML_test)

        # Model evaluation
        evaluator = Evaluator(X_ML_train_scaled, y_ML_train, X_ML_test_scaled, y_test, dl_model="csfddl", dataset="nci-full")
        results_logistic_reg = evaluator.predict_logistic_regression()
        print(results_logistic_reg)

        results_gradient_boosting = evaluator.predict_gradient_boosting()
        print(results_gradient_boosting)
        
data_loader = GraphDataLoader()
wl_csfddl = WL_CSFDDL(data_loader)
wl_csfddl.run()