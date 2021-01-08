"""
Assignment 2, COMP338 - Step 4.2 Compute and report the overall and classification errors per class
Assignment 2, COMP338 - Step 4.3 Compute and show the confusion matrix and analyze the results

Thepnathi Chindalaksanaloet, 201123978
Robert Szafarczyk, 201307211
"""

import numpy as np
import torch as th
import cv2 as cv2
from constants import Constants, Plot_Tools, load_trained_models

class Class_Accuracy(object):
    def __init__(self, net, classes=Constants.CLASSES):
        self.net = net
        self.classes = classes
        self.num_class = len(self.classes)

    def showImg(self, image, predicted):
        window_name = f'Predicted: {Constants.CLASSES[predicted]}'
        cv2.imshow(window_name, image) 
        cv2.waitKey(0)

    # Takes quite long to compute for one model. Might be good idea to use cuda for this part
    def compute_confusion_matrix_for_class_accuracy(self, dataset, batch_size=Constants.default_batch_size):
        # confusion matrix (real label, predicted label)
        confusion_matrix = np.zeros((self.num_class, self.num_class), dtype=np.int64)

        for i in range(len(dataset)):
            image, label = th.tensor([dataset[i]['imNorm']]), th.tensor([dataset[i]['label']])
            image, label = image.to(Constants.device), label.to(Constants.device)
            result = self.net(image)
            _, predicted = th.max(result.data, 1)
            # print(Constants.CLASSES[label])
            # self.showImg(dataset[i]['im'], predicted)
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
    # dictionary that contains the trained cnn model, where key is the model's learning rate
    trained_models = load_trained_models()

    # dictionary to stores the confusion matrix of each model, where key is the learning rates
    confusion_matrix_by_learning_rates = {}

    # Iterate through each model, compute and stores the confusion matrix.
    for num_epochs in Constants.num_epochs:
        confusion_matrix_by_learning_rates[num_epochs] = {}
        for rate in Constants.learning_rates:
            print(f'Learning rate: {rate}, Number of epochs: {num_epochs}')

            class_acc = Class_Accuracy(trained_models[rate], Constants.CLASSES)
            confusion_matrix_by_learning_rates[num_epochs][rate] = \
                class_acc.compute_confusion_matrix_for_class_accuracy(Constants.test_dataset)

            print(Constants.line + "\n")


    # For each model, use matplotlib to display the confusion matrix
    for num_epochs in Constants.num_epochs:
        for rate in Constants.learning_rates:
            print(confusion_matrix_by_learning_rates[num_epochs][rate])
            print(Constants.line)

            Plot_Tools.plot_confusion_matrix(cm=confusion_matrix_by_learning_rates[num_epochs][rate],
                                            classes=Constants.CLASSES,
                                            normalize=False,
                                            title=f'Confusion matrix of CCN model with learning rate of {rate} and number of epochs of {num_epochs}')
        print("\n")

