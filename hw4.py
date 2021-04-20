from abc import ABC

import torch
import torchvision.datasets as datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np

# pytorch training loop
# data loading
# normal CNN model - simple
# - a few convolution layers, some linear layers
# wrap in data parallel
# try with a few different ranks, keep track of time and epoch accuracy for each over 30 epochs
# save accuracy as it goes as numpy arrays


def load_data():
    mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=None)
    mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=None)

    batch_size = 100

    train_loader = torch.utils.data.DataLoader(
        dataset=mnist_trainset,
        batch_size=batch_size,
        shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        dataset=mnist_testset,
        batch_size=batch_size,
        shuffle=False)

    return train_loader, test_loader


def define_model(n_classes):
    model = torch.nn.Sequential(
        nn.Conv2d(6, 3, 20),
        nn.ReLU(),
        nn.Conv2d(3, 1, 10),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(100, 50),
        nn.Linear(50, n_classes)
    )
    return model


class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(6, 3, 20)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(3, 1, 10)
        self.fc1 = nn.Linear(10 * 3, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def run_model():
    train_loader, test_loader = load_data()

    net = ConvNet()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(2):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print('Finished Training')


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # train_loader, test_loader = load_data()
    # print(train_loader)
    run_model()


