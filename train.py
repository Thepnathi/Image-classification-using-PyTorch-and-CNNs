"""
Assignment 2, COMP338 - Step 1. Define a Convolutional Neural Network

Thepnathi Chindalaksanaloet, 201123978
Robert Szafarczyk, 201307211
"""

from imgdata import imageDataset, DefaultTrainSet, DefaultTestSet
from cnn import ConvolutionalNetwork, createLossAndOptimizer


train_dataset = imageDataset('data', 'img_list_test.npy')
test_dataset = imageDataset('data', 'img_list_train.npy')

print(len(train_dataset))
print(len(test_dataset))
print(test_dataset[10])
