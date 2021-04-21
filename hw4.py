from abc import ABC

import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
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
    trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
    mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=trans)
    mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=trans)

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
        nn.Conv2d(1, 3, 20),
        nn.ReLU(),
        nn.Conv2d(1, 1, 10),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(100, 50),
        nn.Linear(50, n_classes)
    )
    return model


class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(10, 20, 5)
        self.fc1 = nn.Linear(20 * 4 * 4, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 20 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def run_model(epoch_num):
    train_loader, test_loader = load_data()

    net = ConvNet()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    loses_epcoh = np.zeros(epoch_num)
    for epoch in range(epoch_num):  # loop over the dataset multiple times

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

        # print epoch statistics
        print("epoch: ", epoch, "loss: ", loss.item())
        loses_epcoh[epoch] = loss.item()

        test_accuracy = []
        for i, (data, labels) in enumerate(test_loader):
            # pass data through network
            outputs = net(data)
            _, predicted = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)
            test_accuracy.append((predicted == labels).sum().item() / predicted.size(0))
            accuracy = np.array(test_accuracy)

        print('accuracy: ', np.average(accuracy))

    print('Finished Training')


def run_model_parallel(epoch_num, process_num):
    train_loader, test_loader = load_data()

    net = ConvNet()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    loses_epcoh = np.zeros(epoch_num)
    for epoch in range(epoch_num):  # loop over the dataset multiple times

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

        # print epoch statistics
        print("epoch: ", epoch, "loss: ", loss.item())
        loses_epcoh[epoch] = loss.item()

    print('Finished Training')


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # train_loader, test_loader = load_data()
    # print(train_loader)
    run_model(5)
