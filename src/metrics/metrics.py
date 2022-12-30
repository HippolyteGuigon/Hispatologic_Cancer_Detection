# To do: Créer une fonction accuracy qui prend en entrée y_true, y_pred et renvoie l'accuracy d'un modèle
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn import metrics

class metrics:
    """
    The goal of this class is to compute, given
    the predicted labels and the true labels, the
    different metrics associated to the prediction
    model
    """

    def __init__(self, y_true, y_pred):
        self.y_true = y_true
        self.y_pred = y_pred

    def accuracy(self) -> float:
        """
        The goal of this function is to
        calculate the accuracy of the model

        Arguments:
            None

        Returns:
            -accuracy: float: The accuracy of the
            model
        """

        correct_pred = np.sum(self.y_pred == self.y_true)
        total_cases = len(self.y_pred)
        accuracy = np.round(correct_pred / total_cases, 3)

        return accuracy

    def precision(self) -> float:
        pass

    def recall(self) -> float:
        pass

    def f1_score(self) -> float:
        pass

    def auc_score(self) -> float:
        pass

    def confusion_matrix(self)->np.array:
        """
        The goal of this function is to compute the confusion
        matrix 
        
        Arguments:
            None
            
        Returns:
            None
        """
        cm = confusion_matrix(self.y_true,self.y_pred)
        return cm

    def plot_confusion_matrix(self):
        """
        The goal of this function is to plot the confusion 
        matrix and to save it once
        
        Arguments:
            None 
            
        Returns:
            None
        """
        confusion_matrix = metrics.confusion_matrix(self.y_true, self.y_pred)
        cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])

