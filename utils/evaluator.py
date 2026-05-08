from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import GradientBoostingClassifier
import matplotlib.pyplot as plt
import seaborn as sns

class Evaluator:
    def __init__(self, X_train, y_train, X_test, y_test, random_state=42):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        # self.split_data()

    # def split_data(self):
    #     self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)


    def predict_logistic_regression(self):
        print("Predicting with Logistic Regression")
        model = LogisticRegression(random_state=0).fit(self.X_train, self.y_train)
        y_hat = model.predict_proba(self.X_test)[:, 1]
        y_pred = model.predict(self.X_test)
        cm = confusion_matrix(self.y_test, y_pred)
        auc = roc_auc_score(self.y_test, y_hat)

        # Visualize confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Class 0', 'Class 1'],
                    yticklabels=['Class 0', 'Class 1'])
        plt.title('Confusion Matrix - With Overfit Protection - Logistic Regression\nAUC: {:.4f}'.format(auc))
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('./cm/confusion_matrix_logistic_regression.png', dpi=300, bbox_inches='tight')
        plt.show()
        return 'AUC: {:.4f}'.format(auc)

    def predict_gradient_boosting(self):
        print("Predicting with Gradient Boosting")
        model = GradientBoostingClassifier(random_state=0).fit(self.X_train, self.y_train)
        y_hat = model.predict_proba(self.X_test)[:, 1]
        y_pred = model.predict(self.X_test)
        cm = confusion_matrix(self.y_test, y_pred)
        auc = roc_auc_score(self.y_test, y_hat)

        # Visualize confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Class 0', 'Class 1'],
                    yticklabels=['Class 0', 'Class 1'])
        plt.title('Confusion Matrix - With Overfit Protection - Gradient Boosting\nAUC: {:.4f}'.format(auc))
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('./cm/confusion_matrix_gradient_boosting.png', dpi=300, bbox_inches='tight')
        plt.show()
        return 'AUC: {:.4f}'.format(auc)