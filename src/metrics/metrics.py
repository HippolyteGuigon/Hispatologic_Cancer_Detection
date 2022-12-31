# To do: Créer une fonction accuracy qui prend en entrée y_true, y_pred et renvoie l'accuracy d'un modèle
import numpy as np
import sklearn
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


class metrics:
    """
    The goal of this class is to compute, given
    the predicted labels and the true labels, the
    different metrics associated to the prediction
    model

    Arguments:
        -y_true: np.array: The true labels
        -y_pred: np.array: The labels computed by
        the model
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
        """
        The goal of this function is to compute
        the precision score of the model

        Arguments:
            None

        Returns:
            -precision_score: float: The precision score
            computed
        """

        precision_score = sklearn.metrics.precision_score(self.y_true, self.y_pred)
        return np.round(precision_score, 2)

    def recall(self) -> float:
        """
        The goal of this function is to compute
        the recall score of the model

        Arguments:
            None

        Returns:
            -recall_score=: float: The recall score
            computed of the model
        """

        recall_score = sklearn.metrics.recall_score(self.y_true, self.y_pred)
        return np.round(recall_score, 2)

    def f1_score(self) -> float:
        """
        The goal of this function is to compute the
        f1_score of the model

        Arguments:
            None

        Returns:
            -f1_score: float: The f1_score of the model
            computed
        """

        f1_score = sklearn.metrics.f1_score(self.y_true, self.y_pred)
        return np.round(f1_score, 2)

    def auc_score(self) -> float:
        pass

    def confusion_matrix(self) -> np.array:
        """
        The goal of this function is to compute the confusion
        matrix

        Arguments:
            None

        Returns:
            None
        """
        cm = confusion_matrix(self.y_true, self.y_pred)
        return cm

    def plot_confusion_matrix(self):
        """
        The goal of this function is to plot the confusion
        matrix and to save it once computed

        Arguments:
            None

        Returns:
            None
        """
        confusion_matrix = self.confusion_matrix()
        cm_display = ConfusionMatrixDisplay(
            confusion_matrix=confusion_matrix, display_labels=[False, True]
        )
        cm_display.plot()
