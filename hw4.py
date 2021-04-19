from abc import ABC

import torch
import torchvision.datasets as datasets
import torch.nn as nn

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

    return mnist_trainset, mnist_testset


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




# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    train, test = load_data()
    print(len(train))
    print(len(test))

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
