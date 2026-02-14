from dict_learners.aksvd import AKSVD
from graph_encoders.wl import WL
from utils.load_data import load_data
from utils.evaluator import Evaluator
graphs, y = load_data(name="nci", size=2)

wl = WL(graphs = graphs)
y_embedding = wl.fit()
aksvd = AKSVD(graph_embeddings=y_embedding)
X = aksvd.fit()

evaluator = Evaluator(X, y, test_size=0.2)

results_logistic_reg = evaluator.predict_logistic_regression()
print(results_logistic_reg)
results_gradient_boosting = evaluator.predict_gradient_boosting()
print(results_gradient_boosting)