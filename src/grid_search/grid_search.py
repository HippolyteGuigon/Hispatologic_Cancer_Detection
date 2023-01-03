# To be coded

# https://stackoverflow.com/questions/44260217/hyperparameter-optimization-for-pytorch-model

# https://stackoverflow.com/questions/44260217/hyperparameter-optimization-for-pytorch-model

import torch.optim as optim
import pandas as pd
from ray import tune
import sys
import os
sys.path.insert(0, os.path.join(os.getcwd(), "src/model"))
sys.path.insert(0, os.path.join(os.getcwd(), "src/logs"))
sys.path.insert(0, os.path.join(os.getcwd(), "src/configs"))
from logs import *
from confs import *
from cnn import *

logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

main_params = load_conf("configs/main.yml", include=True)

def train_model(config)->None:
  """
  The goal of this function is to launch the Grid Search 
  on the CNN model 

  Arguments:
    None 

  Returns:
    None
  """
  #A réécrire (le path)
  os.chdir("/Users/hippodouche/Desktop/histopathologic-cancer-detection")
  model=ConvNeuralNet(main_params["num_classes"],learning_rate=config["learning_rate"],weight_decay=config["weight_decay"])
  model.fit()
  acc=model.evaluate()
  tune.report(accuracy=acc)

if __name__=="__main__":
    main()
    analysis = tune.run(
  train_model, config={"learning_rate": tune.grid_search([0.001, 0.0001, 0.005]),"weight_decay":tune.grid_search([0.005,0.01]),
  "betas": tune.grid_search([(0.5, 0.5), (0.9, 0.999), (0.3, 0.8)])})
    df = analysis.dataframe()
    df.to_csv("result_analysis_gridsearch.csv")
    logging.warning("The Grid Search has just finished running !")
    print("Best config: ", analysis.get_best_config(metric="accuracy"))