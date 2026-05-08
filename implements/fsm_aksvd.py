from dict_learners.aksvd import AKSVD
from graph_encoders.fsm import FSM
from utils.load_data import load_data
from utils.evaluator import Evaluator
from sklearn.preprocessing import StandardScaler

# 1. Load the data
graphs, y = load_data(name="reddit", size=2)

# 2. Instantiate and run the FSM Encoder
# radius=1 extracts 1-hop subgraphs to prevent combinatorial explosion
fsm = FSM(graphs=graphs, radius=1, n_vocab=1000)
y_embedding = fsm.fit()

# Print the top subgraphs to verify the physical structures the algorithm found
print("\nTop 5 Frequent Subgraphs Identified:")
for shape, count in fsm.vocab[:5]:
    print(f"- Topology {shape} appeared {count} times")

# 3. Feed the sparse feature vectors (Y matrix) into the dictionary learner
aksvd = AKSVD(graph_embeddings=y_embedding)
X = aksvd.fit()

# 4. Scale the KSVD Output
# This normalizes the sparse coefficients, ensuring the logistic regression
# optimizer (lbfgs) converges successfully without throwing warnings.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 5. Evaluate the classification performance using the scaled data
evaluator = Evaluator(X_scaled, y, test_size=0.2)
results_logistic_reg = evaluator.predict_logistic_regression()

print("\nClassification Results:")
print(results_logistic_reg)