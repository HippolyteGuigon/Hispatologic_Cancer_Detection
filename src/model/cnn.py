import torch
import torch.nn as nn
import torchvision
from torch.utils.data import random_split
import sys
import os
from PIL import Image
from tqdm import tqdm
import random
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.getcwd(), "src/configs"))
sys.path.insert(0, os.path.join(os.getcwd(), "src/model_save_load"))
sys.path.insert(0, os.path.join(os.getcwd(), "src/transforms"))
from confs import *
from model_save_load import *
from transform import transform

tqdm.pandas()

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

        self.data = torchvision.datasets.ImageFolder(
            root="train", transform=all_transforms
        )
        self.n = len(self.data)

        super(ConvNeuralNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=(1, 1)),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.MaxPool2d(2, 2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=(1, 1)),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.MaxPool2d(2, 2)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=(1, 1)),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.MaxPool2d(2, 2)
        )

        self.fc = nn.Sequential(
             nn.Flatten(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes),
            nn.Sigmoid()
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
        x=self.conv1(x)
        x=self.conv2(x)
        x=self.conv3(x)
        x=self.fc(x)
        return x

    def fit(self):
        """
        The goal of this function is to launch the
        training of the neural network.

        Arguments:
            None

        Returns:
            None
        """

        p = main_params["pipeline_params"]["train_size"]
        train_set, test_set = random_split(
            self.data,
            (int(p * len(self.data)), len(self.data) - int(p * len(self.data))),
        )

        self.train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=batch_size, shuffle=True
        )
        self.test_loader = torch.utils.data.DataLoader(
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
            for i, (images, labels) in enumerate(self.train_loader):
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

    def get_train_test_loader(self) -> torch:
        """
        The goal of this function is to return the
        train and test loader when they've been built

        Arguments:
            None

        Returns:
            -train_loader: torch: The train set
            -test_loader: torch: The test set
        """

        return self.train_loader, self.test_loader

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
            for images, labels in self.test_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                print(predicted)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            print(
                "Accuracy of the network on the {} train images: {} %".format(
                    self.n, 100 * correct / total
                )
            )

    def predict(self, image_path: str) -> int:
        """
        The goal of this function is, after having received an image,
        to predict the associated label

        Arguments:
            -image_path: str: The path of the
            image which label has to be predicted

        Returns:
            -label: int: The predicted label of the image
        """

        model = load_model()
        transformer = transform()
        image = Image.open(image_path)
        input = transformer(image)
        input = input.view(1, 3, 32, 32)
        output = model(input)
        _, predicted = torch.max(output.data, 1)
        return predicted

    def global_predict(self, df=pd.read_csv("train_labels.csv")) -> np.array:
        """
        The goal of this function is, given a global
        DataFrame, to predict the labels for each image
        and to return an array with the predictions

        Arguments:
            df: pd.DataFrame: The DataFrame with images
            to be predicted

        Returns:
            y_pred: np.array: The array with the predicted
            labels
        """

        df = df.loc[:50, :]
        df["id"] = df["id"].apply(lambda x: x + str(".tif"))

        def predict_label(image_name: str) -> str:
            if os.path.exists(os.path.join("train/cancerous", image_name)):
                path = os.path.join("train/cancerous", image_name)
            elif os.path.exists(os.path.join("train/non_cancerous", image_name)):
                path = os.path.join("train/non_cancerous", image_name)
            else:
                path = os.path.join("test", image_name)
            try:
                label = self.predict(path)
            except:
                label = "Unknown"

            return label

        df["prediction"] = df["id"].progress_apply(
            lambda image: predict_label(image).item()
        )
        y_pred = np.array(df["prediction"])
        y_pred = y_pred[y_pred != "Unknown"]
        np.save("lets_check.npy", y_pred)
        print(np.mean(y_pred))
        return y_pred

if __name__ == "__main__":
    model = ConvNeuralNet(2)
    model.global_predict()
