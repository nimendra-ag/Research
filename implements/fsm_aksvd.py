from dict_learners.aksvd import AKSVD
from graph_encoders.fsm import FSM
from utils.graph_data import GraphDataLoader
from utils.evaluator import Evaluator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler
from graph_encoders.fsm_hybrid import FSM_Hybrid

class FSM_AKSVD:
    def __init__(self, data_loader):
        self.implementation = "FSM_AKSVD"
        self.data_loader = data_loader

    def run(self):
        # 1. Load graph data
        graphs, y = self.data_loader.nci_full_graphs, self.data_loader.nci_full_labels

        # 2. Divide the data into train and test sets (Overfit protection step 1)
        G_train, G_test, y_train, y_test = train_test_split(graphs, y, test_size=0.2, random_state=42)

        # 3. Divide train set further into vocab training and ML training sets (Overfit protection step 2)
        G_vocab_train, G_ML_train, y_vocab_train, y_ML_train = train_test_split(G_train, y_train, test_size=0.75,
                                                                                random_state=42)

        # 4. FSM Vocabulary Training
        #fsm = FSM(radius=1, n_vocab=1000)
        fsm = FSM_Hybrid(radius=1,n_vocab=1000)
        graph_embeddings = fsm.generate_training_embeddings(G_vocab_train)

        print("\nTop 5 Frequent Subgraphs Identified (From Vocab Training Set):")
        for shape, count in fsm.vocab[:5]:
            print(f"- Topology {shape} appeared {count} times")

        # 5. Dictionary Learning
        aksvd = AKSVD().fit(training_graph_embeddings=graph_embeddings)

        # 6. Generate Inference Embeddings (ML Train & Test)
        graph_embeddings_ml_train = fsm.generate_inferencing_embeddings(G_ML_train)
        X_ML_train = aksvd.infer(graph_embeddings_ml_train)

        graph_embeddings_ml_test = fsm.generate_inferencing_embeddings(G_test)
        X_ML_test = aksvd.infer(graph_embeddings_ml_test)

        # 7. Scale Output
        scaler = MaxAbsScaler()
        X_ML_train_scaled = scaler.fit_transform(X_ML_train)
        X_ML_test_scaled = scaler.transform(X_ML_test)

        # 8. Model Evaluation
        evaluator = Evaluator(X_ML_train_scaled, y_ML_train, X_ML_test_scaled, y_test)

        print("\nLogistic Regression Results:")
        results_logistic_reg = evaluator.predict_logistic_regression()
        print(results_logistic_reg)

        print("\nGradient Boosting Results:")
        results_gradient_boosting = evaluator.predict_gradient_boosting()
        print(results_gradient_boosting)

# Execution
data_loader = GraphDataLoader()
fsm_ksvd = FSM_AKSVD(data_loader)
fsm_ksvd.run()