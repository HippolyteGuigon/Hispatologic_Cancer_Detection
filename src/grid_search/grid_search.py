# To be coded

# https://stackoverflow.com/questions/44260217/hyperparameter-optimization-for-pytorch-model

# import torch.optim as optim
# from ray import tune
# from ray.tune.examples.mnist_pytorch import get_data_loaders, ConvNet, train, test


# def train_mnist(config):
#    train_loader, test_loader = get_data_loaders()
#    model = ConvNet()
#    optimizer = optim.SGD(model.parameters(), lr=config["lr"])
#    for i in range(10):
#        train(model, optimizer, train_loader)
#        acc = test(model, test_loader)
#        tune.report(mean_accuracy=acc)


# analysis = tune.run(
#   train_mnist, config={"lr": tune.grid_search([0.001, 0.01, 0.1])})

# print("Best config: ", analysis.get_best_config(metric="mean_accuracy"))

# Get a dataframe for analyzing trial results.
# df = analysis.dataframe()
