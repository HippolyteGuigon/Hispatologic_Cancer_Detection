import numpy as np


class EarlyStopper:
    """
    The goal of this class is to implement the early stopping
    algorithm to avoid overfitting if performance on test set
    decreases too much
    """

    def __init__(self, patience=1, min_delta=0):
        """
        The goal of this function is to initialize
        parameters

        Arguments:
            -patience: int: The number of epoch to wait
            after decrease is noticed before stopping training
            -min_delta; float: The tolerance in the difference
            between test loss and minimum test loss

        Returns:
            None
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_test_loss = np.inf

    def early_stop(self, test_loss):
        """
        The goal of this function is to apply the
        early stopping algorithm

        Arguments:
            -test_loss: The loss on the test set computed
            at a certain epoch

        Returns:
            -bool: True if the performance has been decreasing
            over a certain number of epoch
        """
        if test_loss < self.min_test_loss:
            self.min_test_loss = test_loss
            self.counter = 0
        elif test_loss > self.min_test_loss + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                return True

        return False
