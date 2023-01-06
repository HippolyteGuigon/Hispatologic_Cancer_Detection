# The goal of this file is to save the model and load it for further utilisation

import torch


def save_model(model, path="Hispatologic_cancer_detection/model_save_load/model_save.pt") -> None:
    """
    The goal of this function is to save the model once
    it has been trained.

    Arguments:
        -model: torch: The model that has just been trained
        -path: str: The path where the model is to be saved

    Returns:
        None
    """
    torch.save(model, path)


def load_model(path="Hispatologic_cancer_detection/model_save_load/model_save.pt") -> torch:
    """
    The goal of this function is to load the saved model

    Arguments:
        -path: str: The path of the trained model

    Returns:
        -model: torch: The saved model
    """

    model = torch.load(path)

    return model
