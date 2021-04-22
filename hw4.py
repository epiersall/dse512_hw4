from abc import ABC

import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
import torch.distributed as dist

import numpy as np
import pandas as pd
import mpi4py.MPI as MPI


# pytorch training loop
# data loading
# normal CNN model - simple
# - a few convolution layers, some linear layers
# wrap in data parallel
# try with a few different ranks, keep track of time and epoch accuracy for each over 30 epochs
# save accuracy as it goes as numpy arrays


def load_dataset():
    train_dataset = datasets.MNIST(root='./data', train=True, transform=transforms.Compose([transforms.ToTensor(),]),
                                   download=True)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True)

    return train_dataset, test_dataset


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


def train(net, epoch_num, train_loader, test_loader, criterion, optimizer):
    # criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    loses_epoch = np.zeros(epoch_num)
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
        loses_epoch[epoch] = loss.item()

        test_accuracy = []
        for i, (data, labels) in enumerate(test_loader):
            # pass data through network
            outputs = net(data)
            _, predicted = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)
            test_accuracy.append((predicted == labels).sum().item() / predicted.size(0))
            accuracy = np.array(test_accuracy)

        accuracy_av = pd.DataFrame(np.average(accuracy))
        print('accuracy: ', np.average(accuracy))
        accuracy_av.to_csv('accuracy.csv')


def run_model():
    epoch_num = 2
    train_loader, test_loader = load_data()

    net = ConvNet()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    loses_epoch = np.zeros(epoch_num)
    accuracy = []
    for epoch in range(epoch_num):  # loop over the dataset multiple times


        correct = 0
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
            correct += (outputs == labels).float().sum()

        # print epoch statistics
        print("epoch: ", epoch, "loss: ", loss.item())
        loses_epoch[epoch] = loss.item()

        accuracy = correct / len(train_loader)
        print('accuracy: ', accuracy)

    accuracy_df = pd.DataFrame(np.average(accuracy))
    accuracy_df.to_csv('accuracy.csv')

    print('Finished Training')



def run_model_parallel(epoch_num, batch_size, learning_rate=0.05):

    train_data, test_data = load_dataset()

    sampler = DistributedSampler(train_data)
    train_loader = DataLoader(train_data, batch_size=batch_size, sampler=sampler)
    test_loader = DataLoader(test_data, batch_size=batch_size, sampler=sampler)

    model = ConvNet()
    model = DistributedDataParallel(model)

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    learning_rate *= world_size
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    train(model, epoch_num, train_loader, test_loader, criterion, optimizer)



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    rank = MPI.COMM_WORLD.Get_rank()
    world_size = MPI.COMM_WORLD.Get_size()

    if rank == 0:
        print("World size: ", world_size)

    dist.init_process_group('gloo', init_method='env://', world_size=world_size, rank=rank)


    run_model_parallel(2, 30)


