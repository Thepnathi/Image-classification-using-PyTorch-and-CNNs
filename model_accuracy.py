import torch as th
from cnn import ConvolutionalNetwork
from train_network import train_dataset, test_dataset

device = torch.device("cuda" if torch.cuda.is_available() else "gpu")

class Model_Accuracy(object):
    def __init__(self, net):
        self.net = net

    def dataset_accuracy(self, data_loader, name=""):
        net = self.net.to(device)
        correct_prediction = 0
        total = 0

        for i in range(len(data_loader)):
          input = th.tensor([data_loader[i]['imNorm']])
          label = th.tensor([data_loader[i]['label']])
          input, label = input.to(device), label.to(device)
          outputs = self.net(input)
          _, predicted = th.max(outputs, 1)
          total += label.size(0)
          correct_prediction += (predicted == labels).sum()
        accuracy = 100 * float(correct_prediction) / total
        print('Accuracy of the network on the {} {} images: {:2f} %'.format(total, name, accuracy))

    def train_set_accuracy(self, train_loader):
        self.dataset_accuracy(train_loader, "train")

    def val_set_accuracy(self, val_loader):
        self.dataset_accuracy(val_loader, "validation")

    def test_set_accuracy(self, test_loader):
        self.dataset_accuracy(test_loader, "test")

    def compute_accuracy(self, train_loader, test_loader):
        self.train_set_accuracy(train_loader)
        # self.val_set_accuracy(val_loader)
        self.test_set_accuracy(test_loader)

net = ConvolutionalNetwork()
model = Model_Accuracy(net)
train_dataset[0]['label']
model.compute_accuracy(train_dataset, test_dataset)