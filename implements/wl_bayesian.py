from dict_learners.bayesian_dl import BAYESIAN_DL
from graph_encoders.wl import WL
from utils.graph_data import GraphDataLoader
from utils.evaluator import Evaluator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler

# data_loader = GraphDataLoader()

#-------------------------Without overfit protection-------------------------#
# # load graph data
# graphs, y = load_data(name="nci", size=2)
#
# wl = WL(graphs = graphs)
# wl.fit()
# graph_embeddings = wl.get_embeddings()
# bayesian_dl = BayesianDL(graph_embeddings=graph_embeddings)
# X = bayesian_dl.fit()
#
# evaluator = Evaluator(X, y, test_size=0.2)
#
# results_logistic_reg = evaluator.predict_logistic_regression()
# print(results_logistic_reg)
# results_gradient_boosting = evaluator.predict_gradient_boosting()
# print(results_gradient_boosting)

#-------------------------With overfit protection-------------------------#
# # load graph data
# graphs, y = data_loader.nci_full_graphs, data_loader.nci_full_labels
# # First divide the data into train and test sets.
# G_train, G_test, y_train, y_test = train_test_split(graphs, y, test_size=0.2, random_state=42)
#
# # divide the train set further into vocab training and ML training sets
# G_vocab_train, G_ML_train, y_vocab_train, y_ML_train = train_test_split(G_train, y_train, test_size=0.75, random_state=42)
#
# wl = WL()
# graph_embeddings = wl.generate_training_embeddings(G_vocab_train)
#
# bayesian_dl = BAYESIAN_DL().fit(training_graph_embeddings=graph_embeddings)
#
# graph_embeddings_ml_train = wl.generate_inferencing_embeddings(G_ML_train)
# X_ML_train = bayesian_dl.infer(graph_embeddings_ml_train)
#
# graph_embeddings_ml_test = wl.generate_inferencing_embeddings(G_test)
# X_ML_test = bayesian_dl.infer(graph_embeddings_ml_test)
#
# scaler = MaxAbsScaler()
# X_ML_train_scaled = scaler.fit_transform(X_ML_train)
# X_ML_test_scaled = scaler.transform(X_ML_test)
#
# # Model evaluation
# evaluator = Evaluator(X_ML_train_scaled, y_ML_train, X_ML_test_scaled, y_test)
# results_logistic_reg = evaluator.predict_logistic_regression()
# print(results_logistic_reg)
#
# results_gradient_boosting = evaluator.predict_gradient_boosting()
# print(results_gradient_boosting)


class WL_BAYESIAN:
    def __init__(self):
        self.implementation = "WL_BAYESIAN"

    def run(self, G_vocab_train, G_ML_train, G_test, y_ML_train, y_test):
        wl = WL()
        graph_embeddings = wl.generate_training_embeddings(G_vocab_train)

        bayesian_dl = BAYESIAN_DL().fit(training_graph_embeddings=graph_embeddings)

        graph_embeddings_ml_train = wl.generate_inferencing_embeddings(G_ML_train)
        X_ML_train = bayesian_dl.infer(graph_embeddings_ml_train)

        graph_embeddings_ml_test = wl.generate_inferencing_embeddings(G_test)
        X_ML_test = bayesian_dl.infer(graph_embeddings_ml_test)

        scaler = MaxAbsScaler()
        X_ML_train_scaled = scaler.fit_transform(X_ML_train)
        X_ML_test_scaled = scaler.transform(X_ML_test)

        # Model evaluation
        evaluator = Evaluator(X_ML_train_scaled, y_ML_train, X_ML_test_scaled, y_test, dl_model="bayesian", dataset="nci-full")
        results_logistic_reg = evaluator.predict_logistic_regression()
        print(results_logistic_reg)

        results_gradient_boosting = evaluator.predict_gradient_boosting()
        print(results_gradient_boosting)