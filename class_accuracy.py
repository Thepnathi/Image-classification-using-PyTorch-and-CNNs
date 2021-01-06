"""
Assignment 2, COMP338 - Step 4.2 Compute and report the overall and classification errors per class

Thepnathi Chindalaksanaloet, 201123978
Robert Szafarczyk, 201307211
"""

import numpy as np
import torch as th
from constants import Constants
from test_network import load_trained_network, load_trained_models_by_learning_rates

class Class_Accuracy(object):
    def __init__(self, net, classes=Constants.CLASSES):
        self.net = net
        self.classes = classes
        self.num_class = len(self.classes)

    def compute_confusion_matrix_for_class_accuracy(self, dataset, batch_size):
        # confusion matrix (real label, predicted label)
        confusion_matrix = np.zeros((self.num_class, self.num_class), dtype=np.int64)

        for i in range(len(dataset)):
            image, label = th.tensor([dataset[i]['imNorm']]), th.tensor([dataset[i]['label']])
            result = self.net(image)
            _, predicted = th.max(result.data, 1)
            for j in range(batch_size):
                confusion_matrix[label[j], predicted[j]] += 1
                label = label[j]

        print("{:<10} {:^10}".format("Class", "Accuracy (%)"))

        for i in range(self.num_class):
            class_total = confusion_matrix[i, :].sum()
            class_correct = confusion_matrix[i, i]
            percentage_correct = 100.0 * float(class_correct) / class_total

            print('{:<10} {:^10.2f}'.format(self.classes[i], percentage_correct))
        
        return confusion_matrix


if __name__ == "__main__":
    trained_models_by_learning_rates = load_trained_models_by_learning_rates()

    model_1 = trained_models_by_learning_rates[Constants.learning_rates[0]]

    class_accuracy = Class_Accuracy(model_1, Constants.CLASSES)
    
    confus_matrix = class_accuracy.compute_confusion_matrix_for_class_accuracy(Constants.train_dataset, Constants.default_batch_size)
    


