import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import random_split
import sys
import os

sys.path.insert(0, os.path.join(os.getcwd(), "src/configs"))

from confs import *

main_params = load_conf("configs/main.yml", include=True)
batch_size = main_params["cnn_params"]["batch_size"]
num_classes = main_params["cnn_params"]["num_classes"]
learning_rate = main_params["cnn_params"]["learning_rate"]
num_epochs = main_params["cnn_params"]["num_epochs"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Use transforms.compose method to reformat images for modeling,
# and save to variable all_transforms for later use
all_transforms = transforms.Compose(
    [transforms.Resize((32, 32)), transforms.ToTensor()]
)

data = torchvision.datasets.ImageFolder(root="train", transform=all_transforms)

n = len(data)
p = main_params["cnn_params"]["train_size"]
train_set, test_set = random_split(
    data, (int(p * len(data)), len(data) - int(p * len(data)))
)


train_loader = torch.utils.data.DataLoader(
    train_set, batch_size=batch_size, shuffle=True
)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True)

# Creating a CNN class
class ConvNeuralNet(nn.Module):
    #  Determine what layers and their order in CNN object
    def __init__(self, num_classes):
        super(ConvNeuralNet, self).__init__()
        self.conv_layer1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3)
        self.conv_layer2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3)
        self.max_pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv_layer3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.conv_layer4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3)
        self.max_pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(1600, 128)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes)

    # Progresses data across layers
    def forward(self, x):
        out = self.conv_layer1(x)
        out = self.conv_layer2(out)
        out = self.max_pool1(out)

        out = self.conv_layer3(out)
        out = self.conv_layer4(out)
        out = self.max_pool2(out)
        out = out.reshape(out.size(0), -1)
        
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        return out


model = ConvNeuralNet(num_classes)

# Set Loss function with criterion
criterion = nn.CrossEntropyLoss()

# Set optimizer with optimizer
optimizer = torch.optim.SGD(
    model.parameters(), lr=learning_rate, weight_decay=0.005, momentum=0.9
)

total_step = len(train_loader)


# We use the pre-defined number of epochs to
# determine how many iterations to train the network on
for epoch in range(num_epochs):
    # Load in the data in batches using the train_loader object
    for i, (images, labels) in enumerate(train_loader):
        # Move tensors to the configured device
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print("Epoch [{}/{}], Loss: {:.4f}".format(epoch + 1, num_epochs, loss.item()))


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
