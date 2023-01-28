import sys
import os

from Hispatologic_cancer_detection.configs.confs import *
from Hispatologic_cancer_detection.model.cnn import *
from Hispatologic_cancer_detection.model_save_load.model_save_load import *
from Hispatologic_cancer_detection.transforms.transform import *
from Hispatologic_cancer_detection.metrics.metrics import *
from Hispatologic_cancer_detection.logs.logs import *
from Hispatologic_cancer_detection.early_stopping.early_stopping import *
from Hispatologic_cancer_detection.model.transformer import *

main_params = load_conf("configs/main.yml", include=True)


def launch_model() -> None:
    """
    The goal of this function is to launch the model
    chosen by the user

    Arguments:
        None

    Returns:
        None
    """
    model_chosen = main_params["model_chosen"]
    if model_chosen == "cnn":
        model = ConvNeuralNet(main_params["num_classes"])
        model.fit()
        model.save()
        model.evaluate()
        
    elif model_chosen=="transformer":
        model = Transformer()
        model.fit()



if __name__ == "__main__":
    launch_model()
