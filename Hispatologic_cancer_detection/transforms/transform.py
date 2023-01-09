import torchvision.transforms as transforms
import sys
import os

from Hispatologic_cancer_detection.visualisation.visualisation import *
from Hispatologic_cancer_detection.configs.confs import *

main_params = load_conf("configs/main.yml", include=True)


def transform(size=main_params["pipeline_params"]["resize"]):
    """
    The goal of this function is to resize the image
    the appropriate way before it is feeded to the model

    Arguments:
        -size: int: The desired size of the image after
        transformation

    Returns:
        -transformer: torch: The transformer
    """

    transformer = transforms.Compose(
        [
            transforms.Resize(size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(25),
            transforms.ToTensor(),
        ]
    )
    return transformer
