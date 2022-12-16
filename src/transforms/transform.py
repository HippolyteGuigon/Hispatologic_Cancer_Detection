import torch 
import torchvision
import torchvision.transforms as transforms
import sys 
import os 
from PIL import Image

sys.path.insert(0,os.path.join(os.getcwd(),"src/configs"))
from confs import *

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

    transformer=transforms.Compose(
    [transforms.Resize(size),
    transforms.ToTensor()]
)
    return transformer

