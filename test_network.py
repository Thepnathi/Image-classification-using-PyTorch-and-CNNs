"""
Assignment 2, COMP338 - Step 3.3 and 4. Test the Convolutional Neural Network

Thepnathi Chindalaksanaloet, 201123978
Robert Szafarczyk, 201307211
"""

import numpy as np
import torch as th

from cnn import ConvolutionalNetwork
from train_network import learning_rates, CLASSES, gen_model_fname



def load_trained_network(net, learning_rate):
    return net.load_state_dict(th.load(gen_model_fname(learning_rate)))


if __name__ == "__main__":
    net = ConvolutionalNetwork()

    for rate in learning_rates:
        net = load_trained_network(net, rate)

        # Do test step 3.3 and 4.