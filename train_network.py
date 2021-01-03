"""
Assignment 2, COMP338 - Step 3. Train the Convolutional Neural Network

Thepnathi Chindalaksanaloet, 201123978
Robert Szafarczyk, 201307211
"""

import time
import numpy as np
import torch as th

from imgdata import imageDataset, DefaultTrainSet, DefaultTestSet
from cnn import ConvolutionalNetwork, createLossAndOptimizer


train_dataset = imageDataset('data', 'img_list_train.npy')
test_dataset = imageDataset('data', 'img_list_test.npy')

n_training_samples = len(train_dataset)
n_test_samples = len(test_dataset)
# Random indices
train_sampler = th.utils.data.sampler.SubsetRandomSampler(np.arange(n_training_samples, dtype=np.int64))
test_sampler = th.utils.data.sampler.SubsetRandomSampler(np.arange(n_test_samples, dtype=np.int64))

CLASSES = ["airplanes", "cars", "dog", "faces", "keyboard"]
learning_rates = [1e-02, 1e-03, 1e-04, 1e-05]

def gen_model_fname(learning_rate):
    return "model_learning_rate_" + "{:.0e}".format(learning_rate) + ".pth"


def train(net, batch_size, n_epochs, learning_rate):
    """
    Train a neural network, print statistics of the training and save the trained model to
    the file 'model_learning_rate_{learning_rate}.pth'

    Return the training history.

    :param net: (PyTorch Neural Network)
    :param batch_size: (int)
    :param n_epochs: (int)  Number of iterations on the training set
    :param learning_rate: (float) learning rate used by the optimizer
    """
    print("===== HYPERPARAMETERS =====")
    print("batch_size=", batch_size)
    print("n_epochs=", n_epochs)
    print("learning_rate=", learning_rate)
    print("=" * 30)

    criterion, optimizer = createLossAndOptimizer(net, learning_rate)

    # Init variables used for plotting the loss
    train_history = []

    training_start_time = time.time()
    model_fname = gen_model_fname(learning_rate)

    n_minibatches = len(train_dataset) // batch_size
    for epoch in range(n_epochs):  # loop over the dataset multiple times
        start_time = time.time()

        running_loss = 0.0
        total_train_loss = 0

        for i in range(n_minibatches):

            # Gather data for this mini batch
            inputs = th.tensor([train_dataset[j]['imNorm'] for j in range(i, i+batch_size)], dtype=th.float32)
            labels = th.tensor([train_dataset[j]['label'] for j in range(i, i+batch_size)], dtype=th.int64)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            total_train_loss += loss.item()

        train_history.append(total_train_loss / len(train_dataset))
        th.save(net.state_dict(), model_fname)

        print(f'Epoch: {epoch + 1}\trunning_loss: {running_loss}\ttook: {time.time() - start_time}s')
        running_loss = 0.0


    print("Training Finished, took {:.2f}s".format(time.time() - training_start_time))

    # Load the trained model into network
    net.load_state_dict(th.load(model_fname))

    return train_history


if  __name__ == "__main__":
    for rate in learning_rates:
        train(ConvolutionalNetwork(), batch_size=16, n_epochs=20, learning_rate=rate)
