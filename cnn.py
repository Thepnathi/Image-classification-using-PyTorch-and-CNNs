"""
Assignment 2, COMP338 - Step 1. Define a Convolutional Neural Network

Thepnathi Chindalaksanaloet, 201123978
Robert Szafarczyk, 201307211
"""

import torch.nn as nn
import torch.nn.functional as F

class SimpleConvolutionalNetwork(nn.Module):
    def __init__(self):
        super(SimpleConvolutionalNetwork, self).__init__()

        # First hidden layer: a convolution layer with a filter size 7x7, stride 2, padding 3,
        # the number of channels 64, followed by Batch Normalization and ReLu.
        self.hidden1 = F.relu(nn.BatchNorm2d(nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)))

        # Second hidden layer: max pooling with a filter size 3x3, stride 2, padding 0;
        self.hidden2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)

        # Third hidden layer: a convolution layer with a filter size 3x3, stride 1, padding 1,
        # the number of channels 64, followed by Batch Normalization and ReLu.
        self.hidden3 = F.relu(nn.BatchNorm2d(nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)))

        # Fourth hidden layer: max pooling with a filter size 3x3, stride 2, padding 0;
        self.hidden4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)

        # Fully connected layer, with the output channel 5 (i.e., the number of classes);
        self.fc = nn.Linear(57600, 5)


    def forward(self, x):
        """
        Forward pass,
        x shape is (batch_size, 3, 250, 250)
        the comments ignore the batch size (it stays the same accross all layers)
        """
        # When applying a kernel, the shape changes: (N - F + 2P)/S + 1

        # shape : 3x250x250 -> 64x128x128 = (250 - 7 + 2*6)/2 + 1 = 128
        x = self.hidden1(x)

        # 64x128x128 -> 64x62x62
        x = self.hidden2(x)

        # 64x62x62 -> 64x62x62
        x = self.hidden3(x)

        # 64x62x62 -> 64x30x30
        x = self.hidden4(x)

        # Reshape data for the fully connected layer.
        # 64x30x30 -> 57600
        x = x.view(-1, 64 * 30 * 30)
        # 57600 -> 5
        x = self.fc(x)

        # Softmax function to transform the output from the fully connected layer into probabilities.
        probabilities = nn.Softmax(x)

        return probabilities
