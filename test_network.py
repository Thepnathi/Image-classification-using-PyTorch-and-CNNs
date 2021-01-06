"""
Assignment 2, COMP338 - Step 3.3 and 4. Test the Convolutional Neural Network

Thepnathi Chindalaksanaloet, 201123978
Robert Szafarczyk, 201307211
"""

import numpy as np
import torch as th

from cnn import ConvolutionalNetwork
from constants import Constants
from train_network import gen_model_fname

def load_trained_network(net, learning_rate):
    return net.load_state_dict(th.load(gen_model_fname(learning_rate)))

def load_trained_models_by_learning_rates():
    # Stores all the trained cnn models
    trained_models_by_learning_rates = {}

    # Iterate through learning rates and stores the trained model by learning rate
    for rate in Constants.learning_rates:
        model = ConvolutionalNetwork()
        load_trained_network(model, rate)
        trained_models_by_learning_rates[rate] = model

    return trained_models_by_learning_rates

if __name__ == "__main__":

    for rate in Constants.learning_rates:
        net = ConvolutionalNetwork()
        load_trained_network(net, rate)
        print(net)

        # Do test step 3.3 and 4.