import sys
import os

sys.path.insert(0, os.path.join(os.getcwd(), "src/configs"))
from confs import *
from cnn import *

main_params = load_conf("configs/main.yml", include=True)
model_chosen = main_params["model_chosen"]

if model_chosen == "cnn":
    model = ConvNeuralNet(main_params["num_classes"])
    model.fit()
    model.save()
