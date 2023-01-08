import torch
import torch.nn as nn
import torchvision
from torch.utils.data import random_split
import sys
import os
from PIL import Image
from tqdm import tqdm
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")

from Hispatologic_cancer_detection.configs.confs import *
from Hispatologic_cancer_detection.model_save_load.model_save_load import *
from Hispatologic_cancer_detection.transforms.transform import *
from Hispatologic_cancer_detection.metrics.metrics import *
from Hispatologic_cancer_detection.logs.logs import *
from Hispatologic_cancer_detection.early_stopping.early_stopping import *

logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

tqdm.pandas()

main_params = load_conf("configs/main.yml", include=True)
main_params = clean_params(main_params)
batch_size = main_params["batch_size"]
num_classes = main_params["num_classes"]
learning_rate = main_params["learning_rate"]
num_epochs = main_params["num_epochs"]
dropout = main_params["dropout"]
weight_decay = main_params["weight_decay"]
early_stopping = main_params["early_stopping"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Use transforms.compose method to reformat images for modeling,
# and save to variable all_transforms for later use
all_transforms = transform()
early_stopping_model = EarlyStopper(patience=3)

# Creating a CNN class
class ConvNeuralNet(nn.Module):
    """
    The goal of this class is implementing a CNN model
    for the classification of images
    """

    def __init__(self, num_classes: int, **kwargs):
        """
        The goal of this function is initialisation of
        the arguments that will be used inside this class
        Arguments:
            -num_classes: int: The number of classes in the
            classification problem. In this problem, 2
            -kwargs: The different parameters the user can enter. If he
            doesn't, they will be taken in the configs file
        Returns:
            None
        """
        logging.info("Model parameters initialization has begun")

        for param, value in kwargs.items():
            if param not in main_params.keys():
                raise AttributeError(f"The CNN model has no attribute {param}")
            else:
                main_params[param] = value
        self.data = torchvision.datasets.ImageFolder(
            root="train", transform=all_transforms
        )
        self.n = len(self.data)

        p = main_params["train_size"]
        train_set, test_set = random_split(
            self.data,
            (int(p * len(self.data)), len(self.data) - int(p * len(self.data))),
        )
        self.train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=batch_size, shuffle=False
        )

        self.test_loader = torch.utils.data.DataLoader(
            test_set, batch_size=batch_size, shuffle=False
        )

        super(ConvNeuralNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=(1, 1)),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.MaxPool2d(2, 2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=(1, 1)),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.MaxPool2d(2, 2),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=(1, 1)),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.MaxPool2d(2, 2),
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
            nn.Softmax(),
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
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.fc(x)
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

        self.model = ConvNeuralNet(num_classes)

        # Set Loss function with criterion
        criterion = nn.CrossEntropyLoss()

        # Set optimizer with optimizer
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

        # We use the pre-defined number of epochs to
        # determine how many iterations to train the network on
        logging.warning(
            f"Fitting of the model has begun, the params are :{main_params.items()}"
        )
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

            with torch.no_grad():
                correct_test = 0
                total_test = 0
                for images, labels in self.test_loader:
                    images = images.to(device)
                    labels = labels.to(device)
                    outputs = self.model(images)
                    test_loss = criterion(outputs, labels)
                    _, predicted = torch.max(outputs.data, 1)
                    total_test += labels.size(0)
                    correct_test += (predicted == labels).sum().item()

                logging.info(
                    f"Epoch [{epoch + 1}/{num_epochs}], Loss: {np.round(loss.item(),3)}. Accuracy of the network on the test set: {np.round(100 * correct_test / total_test,3)} %"
                )
            if early_stopping and early_stopping_model.early_stop(test_loss):
                logging.warning(
                    f"The model training has stopped at epoch {epoch + 1} due to the early stopping criterion"
                )
                break

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

    def evaluate(self) -> float:
        """
        The goal of this function is, once the model has
        been trained, to evaluate it by printing its
        accuracy
        Arguments:
            None
        Returns:
            -accuracy: float: The accuracy of the model
            computed
        """
        self.model = load_model()
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in self.test_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            logging.info(
                "Accuracy of the network on the {} test images: {} %".format(
                    len(self.test_loader), 100 * correct / total
                )
            )

        accuracy = 100 * correct / total
        return accuracy

    def predict(self, image_path: str, load_model=True) -> int:
        """
        The goal of this function is, after having received an image,
        to predict the associated label
        Arguments:
            -image_path: str: The path of the
            image which label has to be predicted
        Returns:
            -label: int: The predicted label of the image
        """

        if load_model:
            model = load_model()
        else:
            self.fit()
            logging.info("Model has been fitted for prediction")


        transformer = transform()
        image = Image.open(image_path)
        input = transformer(image)
        input = input.view(1, 3, 32, 32)
        if load_model:
            output = model(input)
        else:
            output = self.model(input)
        _, predicted = torch.max(output.data, 1)
        predicted=predicted.item()
        
        return predicted

    def get_pred(self) -> np.array:
        """
        The goal of this function is to get the predictions
        and the actual labels

        Arguments:
            None

        Returns:
            -y_true: np.array: The actual labels
            -y_pred: np.array: The predicted labels
        """

        model = load_model()

        with torch.no_grad():
            y_true = np.array([])
            y_pred = np.array([])
            for images, labels in self.test_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                y_true = np.hstack((y_true, np.array(labels)))
                y_pred = np.hstack((y_pred, np.array(predicted)))
        np.save("y_pred.npy", y_pred)
        np.save("y_true.npy", y_true)

        return y_true, y_pred

    def global_predict(self) -> np.array:
        """
        The goal of this function is, given a global
        DataFrame, to predict the labels for each image
        and to return an array with the predictions
        Arguments:
            -df: pd.DataFrame: The DataFrame with images
            to be predicted
        Returns:
            -y_pred: np.array: The array with the predicted
            labels
            -y_true: np.array: The array with the real labels
        """
        df=pd.read_csv("train_labels.csv")
        df["id"] = df["id"].apply(lambda x: x + str(".tif"))

        def predict_label(image_name: str) -> str:
            if os.path.exists(os.path.join("train/1. cancerous", image_name)):
                path = os.path.join("train/1. cancerous", image_name)
            elif os.path.exists(os.path.join("train/0. non_cancerous", image_name)):
                path = os.path.join("train/0. non_cancerous", image_name)
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
        y_true = np.array(df["label"])
        return y_pred, y_true


if __name__ == "__main__":
    model = ConvNeuralNet(num_classes)
