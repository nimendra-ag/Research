from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


class Evaluator:
    def __init__(self, X, y, test_size=0.2, random_state=42):
        self.X = X
        self.y = y
        self.test_size = test_size
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.split_data()

    def split_data(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)


    def predict_logistic_regression(self):
        print("Predicting with Logistic Regression")
        model = LogisticRegression(random_state=0).fit(self.X_train, self.y_train)
        y_hat = model.predict_proba(self.X_test)[:, 1]
        auc = roc_auc_score(self.y_test, y_hat)
        print('Without overfit protection')
        return 'AUC: {:.4f}'.format(auc)