from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_curve
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import auc
import matplotlib.pyplot as plt

import numpy as np

from sklearn.ensemble import RandomForestClassifier

# 4 Calcular performance de los modelos
#### 4.1 F-Score
#### 4.2 Precision
#### 4.3 Recall
#### 4.4 Accuracy


## Recall --> tp / (tp + fn) --> tp = true positives, fn = false negatives
## Precision --> tp / (tp + fp) --> tp = true positives, fp = false positives
## f1_score --> 2 * (precision * recall) / (precision + recall)
## Accuracy --> (tp + tn) / (tp + tn + fp + fn) --> tp = true positives, tn = true negatives, fp = false positives, fn = false negatives


class Performance:
    def __init__(self, y_true, y_pred):
        self.y_true = y_true
        self.y_pred = y_pred

    
    def display_roc_curve(self, y_true, predictions):
        fpr, tpr, thresholds = roc_curve(y_true, predictions)
        roc_auc = auc(fpr, tpr)
        display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name='estimated value')
        display.plot()
        plt.show()

    def roc_and_pr(self, y_v, probs):
        n_classes = 3
        from sklearn.metrics import f1_score, precision_recall_curve, average_precision_score, roc_curve, auc

        # Compute Precision-Recall and plot curve
        precision = {}
        recall = {}
        average_precision = {}
        plt.figure()
        for i in range(n_classes):
            # get list of values in y_v that are equal to i
            y_true = [1 if y == i else 0 for y in y_v]

            precision[i], recall[i], _ = precision_recall_curve(y_true, probs[:, i])
            average_precision[i] = average_precision_score(y_true, probs[:, i])

            plt.plot(recall[i], precision[i],
            label='Precision-recall curve of class {0} (area = {1:0.2f})'
                                ''.format(i, average_precision[i]))
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.legend(loc="upper right")

            
        # Compute ROC curve and ROC area for each class
        fpr = {}
        tpr = {}
        roc_auc = {}
        for i in range(n_classes):
            y_true = [1 if y == i else 0 for y in y_v]
            fpr[i], tpr[i], _ = roc_curve(y_true, probs[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Compute micro-average ROC curve and ROC area
        # Plot ROC curve
        plt.figure()
        for i in range(n_classes):
            plt.plot(fpr[i], tpr[i], label='ROC curve of class {0} (area = {1:0.2f})' ''.format(i, roc_auc[i]))
        plt.legend()
        plt.show()

    def display_confusion_matrix(self,clf, X_test, y_test):
        from sklearn.metrics import plot_confusion_matrix
        plot_confusion_matrix(clf, X_test, y_test)  
        plt.show()


    def get_performance(self):
        return {
            'f1_score': f1_score(self.y_true, self.y_pred, average='macro'),
            'precision': precision_score(self.y_true, self.y_pred, average='macro'),
            'recall': recall_score(self.y_true, self.y_pred, average='macro'),
            'accuracy': accuracy_score(self.y_true, self.y_pred),
        }
    

    
# y_true = [0, 1, 2, 0, 1, 2]
# y_pred = [0, 2, 1, 0, 0, 1]
# p = Performance()

# a = p.calculate_accuracy_score(y_true, y_pred)
# print(a)

    
    
    