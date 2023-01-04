import torch.optim as optim
import pandas as pd
import ray
from ray import tune
import sys
import os
import numpy as np
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
grid_search_params = load_conf("configs/grid_search.yml")
ray.init(configure_logging=False, log_to_driver=False)
os.system("export PYTHONPATH='$PWD/src/model'")

def train_model(config)->None:
  """
  The goal of this function is to launch the Grid Search 
  on the CNN model 

  Arguments:
    None 

  Returns:
    None
  """

  current_dir_path=os.getcwd()
  while current_dir_path.split("/")[-1] != "histopathologic-cancer-detection":
    current_dir_path=os.path.dirname(current_dir_path)

  os.chdir(current_dir_path)
  model=ConvNeuralNet(main_params["num_classes"],weight_decay=config["weight_decay"])
  model.fit()
  acc=model.evaluate()
  tune.report(accuracy=acc)

if __name__=="__main__":
    main()
    analysis = tune.run(
  train_model, config={"weight_decay":tune.grid_search(grid_search_params["cnn"]["weight_decay_grid"]),
  "learning_rate":tune.grid_search(grid_search_params["cnn"]["learning_rate_grid"]),
  "betas_grid":tune.grid_search(grid_search_params["cnn"]["betas_grid"])})
    df = analysis.dataframe()
    df.to_csv("result_analysis_gridsearch.csv")
    best_result=np.round(df["accuracy"].max(),2)
    logging.warning(f"The Grid Search has just finished running ! The best result is {best_result}%")
