"""
Assignment 2, COMP338 - Step 4.2 Compute and report the overall and classification errors per class

Thepnathi Chindalaksanaloet, 201123978
Robert Szafarczyk, 201307211
"""

import numpy as np
import torch as th

class Class_Accuracy(object):
    def __init__(self, net, classes):
        self.net = net
        self.classes = classes
        self.num_class = len(self.classes)

    def compute_confusion_matrix_for_class_accuracy(self, dataset, batch_size):
        # confusion matrix (real label, predicted label)
        confusion_matrix = np.zeros((self.num_class, self.num_class), dtype=np.int64)

        for i in range(len(dataset)):
            image, label = dataset[i], dataset[i]
            result = self.net(image)
            _, predicted = th.max(result.data, 1)
            for i in range(batch_size):
                confusion_matrix[label[i], predicted[i]] += 1
                label = labels[i]

        print("{:<10} {:^10}".format("Class", "Accuracy (%)"))

        for i in range(self.num_class):
            class_total = confusion_matrix[i, :].sum()
            class_correct = confusion_matrix[i, i]
            percentage_correct = 100.0 * float(class_correct) / class_total

            print('{:<10} {:^10.2f}'.format(self.classes[i], percentage_correct))
        
        return confusion_matrix


if __name__ == "__main__":
    class_acc = Class_Accuracy()


