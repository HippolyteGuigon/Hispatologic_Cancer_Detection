import torch
import torch.nn as nn
import torchvision
from torch.utils.data import random_split
import sys
import os

sys.path.insert(0, os.path.join(os.getcwd(), "src/configs"))
sys.path.insert(0, os.path.join(os.getcwd(), "src/model_save_load"))
sys.path.insert(0, os.path.join(os.getcwd(), "src/transforms"))
from confs import *
from model_save_load import *
from transform import transform

main_params = load_conf("configs/main.yml", include=True)
batch_size = main_params["cnn_params"]["batch_size"]
num_classes = main_params["num_classes"]
learning_rate = main_params["cnn_params"]["learning_rate"]
num_epochs = main_params["cnn_params"]["num_epochs"]
dropout = main_params["cnn_params"]["dropout"]
weight_decay = main_params["cnn_params"]["weight_decay"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Use transforms.compose method to reformat images for modeling,
# and save to variable all_transforms for later use
all_transforms = transform()


# Creating a CNN class
class ConvNeuralNet(nn.Module):
    """
    The goal of this class is implementing a CNN model
    for the classification of images
    """

    def __init__(self, num_classes: int):
        """
        The goal of this function is initialisation of
        the arguments that will be used inside this class

        Arguments:
            -num_classes: The number of classes in the
            classification problem. In this problem, 2

        Returns:
            None
        """

        super(ConvNeuralNet, self).__init__()
        self.network = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=(1, 1)),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=(1, 1)),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=(1, 1)),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(4608, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes),
        )

    # Progresses data across layers
    def forward(self, x):
        """
        The goal of this function is passing on a given image
        throughout the neural network

        Arguments:
            -x: The image to be given to the neural network

        Returns:
            -network: The neural network feeded with the image
        """
        return self.network(x)

    def fit(self):
        """
        The goal of this function is to launch the
        training of the neural network.

        Arguments:
            None

        Returns:
            None
        """

        data = torchvision.datasets.ImageFolder(root="train", transform=all_transforms)

        n = len(data)
        p = main_params["pipeline_params"]["train_size"]
        train_set, test_set = random_split(
            data, (int(p * len(data)), len(data) - int(p * len(data)))
        )

        train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=batch_size, shuffle=True
        )
        test_loader = torch.utils.data.DataLoader(
            test_set, batch_size=batch_size, shuffle=True
        )

        self.model = ConvNeuralNet(num_classes)

        # Set Loss function with criterion
        criterion = nn.CrossEntropyLoss()

        # Set optimizer with optimizer
        optimizer = torch.optim.SGD(
            self.model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

        # We use the pre-defined number of epochs to
        # determine how many iterations to train the network on
        for epoch in range(num_epochs):
            # Load in the data in batches using the train_loader object
            for i, (images, labels) in enumerate(train_loader):
                # Move tensors to the configured device
                images = images.to(device)
                labels = labels.to(device)

                # Forward pass
                outputs = self.model(images)
                loss = criterion(outputs, labels)

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            print(
                "Epoch [{}/{}], Loss: {:.4f}".format(epoch + 1, num_epochs, loss.item())
            )

    def save(self):
        """
        The goal of this function is, once the
        model has been trained, to save it

        Arguments:
            None

        Returns:
            None
        """

        save_model(self.model)

    def evaluate(self):
        """
        The goal of this function is, once the model has
        been trained, to evaluate it by printing its
        accuracy

        Arguments:
            None

        Returns:
            None
        """

        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in train_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            print(
                "Accuracy of the network on the {} train images: {} %".format(
                    n, 100 * correct / total
                )
            )
