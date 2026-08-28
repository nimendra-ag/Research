from dict_learners.aksvd import AKSVD
from graph_encoders.wl import WL
from utils.load_data import load_data
from utils.evaluator import Evaluator
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from WL_KSVD import WL_KSVD
from sklearn.preprocessing import MaxAbsScaler
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

graphs, y = load_data(name="nci1", size=1)

import time

# Start the timer
start_time = time.perf_counter()

#-------------------------------------------------------------------------------


# Over-fit protection
print('Over-fit protection')

G_train, G_test, y_train, y_test = train_test_split(graphs, y, test_size=0.2, random_state=44)

# divide the train set further into vocab training and ML training sets

G_vocab_train, G_ML_train, y_vocab_train, y_ML_train = train_test_split(G_train, y_train, test_size=0.75, random_state=44)

# Fit the WL+KSVD model on the vocab training set
print('Fitting the embedding model')
model = WL_KSVD()
# model = Graph2Vec()

model.fit(G_vocab_train)
X_vocab_train = model.get_embedding()

# Infer the embedding of the ML training set
X_ML_train = model.infer(G_ML_train)
X_ML_test = model.infer(G_test)

# Scaling the embedding such that sparsity is preserved
scaler = MaxAbsScaler()
X_ML_train_scaled = scaler.fit_transform(X_ML_train)

X_ML_test_scaled = scaler.transform(X_ML_test)

# Applying ML model
print('Applying ML model')


print("Predicting with Logistic Regression")
downstream_model_logreg = LogisticRegression(random_state=44).fit(X_ML_train_scaled, y_ML_train)
y_ML_hat = downstream_model_logreg.predict_proba(X_ML_test_scaled)[:, 1]
auc = roc_auc_score(y_test, y_ML_hat)
print('AUC - Logistic Regression: {:.4f}'.format(auc))

print("Predicting with Gradient Boosting")
downstream_model_gb = GradientBoostingClassifier(random_state=44).fit(X_ML_train_scaled, y_ML_train)
y_ML_hat = downstream_model_gb.predict_proba(X_ML_test_scaled)[:, 1]
auc = roc_auc_score(y_test, y_ML_hat)
print('AUC - Gradient Boosting: {:.4f}'.format(auc))


print("Predicting with SVM")
base_model = LinearSVC(random_state=44)
downstream_model_svm = CalibratedClassifierCV().fit(X_ML_train_scaled, y_ML_train)
y_ML_hat = downstream_model_svm.predict_proba(X_ML_test_scaled)[:, 1]
auc = roc_auc_score(y_test, y_ML_hat)
print('AUC - SVM: {:.4f}'.format(auc))


print("Predicting with Random Forest")
downstream_model_rf = RandomForestClassifier(random_state=44).fit(X_ML_train_scaled, y_ML_train)
y_ML_hat = downstream_model_rf.predict_proba(X_ML_test_scaled)[:, 1]
auc = roc_auc_score(y_test, y_ML_hat)
print('AUC - RF: {:.4f}'.format(auc))

# End the timer
end_time = time.perf_counter()

# Calculate duration
duration = end_time - start_time
print(f"Execution time: {duration:.6f} seconds")