# To do: Créer une fonction accuracy qui prend en entrée y_true, y_pred et renvoie l'accuracy d'un modèle


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
        pass

    def precision(self) -> float:
        pass

    def recall(self) -> float:
        pass

    def f1_score(self) -> float:
        pass

    def auc_score(self) -> float:
        pass

    def confusion_matrix(self):
        pass
