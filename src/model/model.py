import sys
import os

sys.path.insert(0, os.path.join(os.getcwd(), "src/configs"))
sys.path.insert(0, os.path.join(os.getcwd(), "src/model"))
sys.path.insert(0, os.path.join(os.getcwd(), "src/transforms"))
from confs import *
from cnn import *

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


if __name__ == "__main__":
    launch_model()
