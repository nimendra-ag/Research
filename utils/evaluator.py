from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score, confusion_matrix
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.utils.class_weight import compute_sample_weight
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
from pathlib import Path

from sklearn.utils import compute_sample_weight

class Evaluator:
    def __init__(
        self, X_train, y_train, X_test, y_test, dl_model, dataset, random_state=42, timestamp=None):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.dl_model = dl_model
        self.dataset = dataset
        self.timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") if timestamp is None else timestamp

    # def split_data(self):
    #     self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)


    def predict_logistic_regression(self):
        print("Predicting with Logistic Regression")
        model = LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000).fit(self.X_train, self.y_train)
        y_hat = model.predict_proba(self.X_test)[:, 1]
        y_pred = model.predict(self.X_test)
        cm = confusion_matrix(self.y_test, y_pred)
        auc = roc_auc_score(self.y_test, y_hat)
        f1_mac = f1_score(self.y_test, y_pred, average='macro')
        pr_auc = average_precision_score(self.y_test, y_hat)

        # Visualize confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Class 0', 'Class 1'],
                    yticklabels=['Class 0', 'Class 1'])
        plt.title('Confusion Matrix - With Overfit Protection - Logistic Regression\nAUC: {:.4f}'.format(auc))
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        path = Path('./cm/{dl_model}_{dataset}_{timestamp}'.format(dl_model=self.dl_model, dataset=self.dataset, timestamp=self.timestamp))
        os.makedirs(path, exist_ok=True)
        plt.savefig(path / 'confusion_matrix_logistic_regression.png', dpi=300, bbox_inches='tight')
        plt.show()
        return 'AUC: {:.4f}, F1: {:.4f}, PR-AUC: {:.4f}'.format(auc, f1_mac, pr_auc)

    def predict_gradient_boosting(self):
        print("Predicting with Gradient Boosting")
        sample_weights = compute_sample_weight('balanced', self.y_train)
        model = GradientBoostingClassifier(random_state=42, n_estimators=100).fit(self.X_train, self.y_train, sample_weight=sample_weights)
        y_hat = model.predict_proba(self.X_test)[:, 1]
        y_pred = model.predict(self.X_test)
        cm = confusion_matrix(self.y_test, y_pred)
        auc = roc_auc_score(self.y_test, y_hat)
        f1_mac = f1_score(self.y_test, y_pred, average='macro')
        pr_auc = average_precision_score(self.y_test, y_hat)

        # Visualize confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Class 0', 'Class 1'],
                    yticklabels=['Class 0', 'Class 1'])
        plt.title('Confusion Matrix - With Overfit Protection - Gradient Boosting\nAUC: {:.4f}'.format(auc))
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        path = Path('./cm/{dl_model}_{dataset}_{timestamp}'.format(dl_model=self.dl_model, dataset=self.dataset, timestamp=self.timestamp))
        os.makedirs(path, exist_ok=True)
        plt.savefig(path / 'confusion_matrix_gradient_boosting.png', dpi=300, bbox_inches='tight')
        plt.show()
        return 'AUC: {:.4f}, F1: {:.4f}, PR-AUC: {:.4f}'.format(auc, f1_mac, pr_auc)