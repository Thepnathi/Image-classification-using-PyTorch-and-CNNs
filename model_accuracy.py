import torch
from cnn import ConvolutionalNetwork

# Test the ndetwork against the test dataset

class Model_Accuracy(object):
    def __init__(self, net):
        self.net = net

    def dataset_accuracy(self, data_loader, name=""):
        net = self.net.to(device)
        correct_prediction = 0
        total = 0
        
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0) 
            correct_prediction += (predicted == labels).sum()
        accuracy = 100 * float(correct_prediction) / total
        print('Accuracy of the network on the {} {} images: {:2f} %'.format(total, name, accuracy))

    def train_set_accuracy(self, train_loader):
        dataset_accuracy(train_loader, "train")

    def val_set_accuracy(self, val_loader):
        dataset_accuracy(val_loader, "validation")

    def test_set_accuracy(self, test_loader):
        dataset_accuracy(test_loader, "test")

    def compute_accuracy(self, train_loader, val_loader, test_loader):
        train_set_accuracy(train_loader)
        val_set_accuracy(val_loader)
        test_set_accuracy(test_loader)

if __name__ == "__main__":
    net = ConvolutionalNetwork()
    model = Model_Accuracy(net)
