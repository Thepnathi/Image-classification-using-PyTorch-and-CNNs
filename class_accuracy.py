"""
Assignment 2, COMP338 - Step 4.2 Compute and report the overall and classification errors per class

Thepnathi Chindalaksanaloet, 201123978
Robert Szafarczyk, 201307211
"""

import numpy as np
import torch as th
from constants import Constants, Plot_Tools
from test_network import load_trained_network, load_trained_models_by_learning_rates

class Class_Accuracy(object):
    def __init__(self, net, classes=Constants.CLASSES):
        self.net = net
        self.classes = classes
        self.num_class = len(self.classes)

    # Takes quite long to compute for one model. Might be good idea to use cuda for this part
    def compute_confusion_matrix_for_class_accuracy(self, dataset, batch_size=Constants.default_batch_size):
        # confusion matrix (real label, predicted label)
        confusion_matrix = np.zeros((self.num_class, self.num_class), dtype=np.int64)


        for i in range(len(dataset)):
            image, label = th.tensor([dataset[i]['imNorm']]), th.tensor([dataset[i]['label']])
            image, label = image.to(Constants.device), label.to(Constants.device)
            result = self.net(image)
            _, predicted = th.max(result.data, 1)
            confusion_matrix[label, predicted] += 1

        print("{:<10} {:^10}".format("Class", "Accuracy (%)"))

        overall_correct = 0
        for i in range(self.num_class):
            class_total = confusion_matrix[i, :].sum()
            class_correct = confusion_matrix[i, i]
            percentage_correct = 100.0 * float(class_correct) / class_total
            overall_correct += percentage_correct

            print('{:<10} {:^10.2f}'.format(self.classes[i], percentage_correct))

        print(f'\nOverall Accuracy: {overall_correct // self.num_class}%')
        
        return confusion_matrix


if __name__ == "__main__":
    trained_models_by_learning_rates = load_trained_models_by_learning_rates()

    confusion_matrix_by_learning_rates = {}

    for rate in Constants.learning_rates:
        class_acc = Class_Accuracy(trained_models_by_learning_rates[rate], Constants.CLASSES)
        confusion_matrix_by_learning_rates[rate] = class_acc.compute_confusion_matrix_for_class_accuracy(Constants.train_dataset)
        print(Constants.line)

    for rate in Constants.learning_rates:
        print(confusion_matrix_by_learning_rates[rate])
        print(Constants.line)
    


