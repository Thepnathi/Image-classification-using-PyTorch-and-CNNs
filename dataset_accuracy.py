from train_network import learning_rates, train_dataset, test_dataset
from test_network import load_trained_network
from cnn import ConvolutionalNetwork
import torch as th

class Dataset_Accuracy(object):
    def __init__(self, net):
        self.net = net

    def dataset_accuracy(self, dataset, name=""):
        correct_prediction = 0
        total = 0

        for i in range(len(dataset)):
          input = th.tensor([dataset[i]['imNorm']])
          label = th.tensor([dataset[i]['label']])
          outputs = self.net(input)
          _, predicted = th.max(outputs, 1)
          total += label.size(0)
          correct_prediction += (predicted == label).sum()
        accuracy = 100 * float(correct_prediction) / total
        print('Accuracy of the network on the {} {} images: {:2f} %'.format(total, name, accuracy))

    def train_dataset_accuracy(self, train_loader):
        self.dataset_accuracy(train_loader, "train")

    def validation_dataset_accuracy(self, validaition_set):
        self.dataset_accuracy(validaition_set, "validation")

    def test_dataset_accuracy(self, test_loader):
        self.dataset_accuracy(test_loader, "test")

    def compute_dataset_accuracy(self, train_dataset=None, test_dataset=None):
        self.train_dataset_accuracy(train_dataset) if train_dataset else None
        self.test_dataset_accuracy(test_dataset) if test_dataset else None
        print("====================")


if __name__ == "__main__":
    trained_models_by_learning_rates = {}

    for rate in learning_rates:
        model = ConvolutionalNetwork()
        load_trained_network(model, rate)
        trained_models_by_learning_rates[rate] = model

    for rate in learning_rates:
        print(f'Learning rate: {rate}')
        loaded_trained_model = trained_models_by_learning_rates[rate]
        dataset_accuracy = Dataset_Accuracy(loaded_trained_model)
        dataset_accuracy .compute_dataset_accuracy(train_dataset, test_dataset)