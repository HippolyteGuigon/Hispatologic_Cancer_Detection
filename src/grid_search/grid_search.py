# To be coded

# https://stackoverflow.com/questions/44260217/hyperparameter-optimization-for-pytorch-model

import torch.optim as optim
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

# from ray.tune.examples.mnist_pytorch import get_data_loaders, ConvNet, train, test

def train_model(config):
    model=ConvNeuralNet(main_params["num_classes"],learning_rate=config["learning_rate"],weight_decay=config["weight_decay"])
    model.fit()
    acc=model.evaluate()
    tune.report(accuracy=acc)
# def train_mnist(config):
#    train_loader, test_loader = get_data_loaders()
#    model = ConvNet()
#    optimizer = optim.SGD(model.parameters(), lr=config["lr"])
#    for i in range(10):
#        train(model, optimizer, train_loader)
#        acc = test(model, test_loader)
#        tune.report(mean_accuracy=acc)




# Get a dataframe for analyzing trial results.
# df = analysis.dataframe()

if __name__=="__main__":
    main()
    analysis = tune.run(
  train_model, config={"learning_rate": tune.grid_search([0.001, 0.0001, 0.005]),"weight_decay":tune.grid_search([0.005,0.01])})
    print("Best config: ", analysis.get_best_config(metric="accuracy"))