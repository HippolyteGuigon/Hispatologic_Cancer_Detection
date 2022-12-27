import torchvision.transforms as transforms
import sys
import os

sys.path.insert(0, os.path.join(os.getcwd(), "src/configs"))
sys.path.insert(0, os.path.join(os.getcwd(), "src/visualisation"))
from confs import *
from visualisation import *

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

    transformer = transforms.Compose([transforms.Resize(size),
    transforms.CenterCrop(size),
    transforms.ToTensor(),
    transforms.Normalize(mean = [0.4296, 0.1155, 0.4047],
                         std= [0.4639, 0.5511, 0.4269])])
    return transformer
