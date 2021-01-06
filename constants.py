import numpy as np
import torch as th
import matplotlib.pyplot as plt
from imgdata import imageDataset, DefaultTrainSet, DefaultTestSet

class Constants(object):
    device = th.device("cuda" if th.cuda.is_available() else "cpu")

    default_batch_size = 16
    default_num_epochs = 20

    train_dataset = imageDataset('data', 'img_list_train.npy')
    test_dataset = imageDataset('data', 'img_list_test.npy')

    n_training_samples = len(train_dataset)
    n_test_samples = len(test_dataset)

    # Random indices
    train_sampler = th.utils.data.sampler.SubsetRandomSampler(np.arange(n_training_samples, dtype=np.int64))
    test_sampler = th.utils.data.sampler.SubsetRandomSampler(np.arange(n_test_samples, dtype=np.int64))

    CLASSES = ["airplanes", "cars", "dog", "faces", "keyboard"]
    learning_rates = [1e-02, 1e-03, 1e-04, 1e-05]

    TRAIN_HISTORY_FNAME = 'train_history_dict.npy'

    line = "=================="


class Plot_Tools(object):
    def plot_confusion_matrix(cm, classes,
                            normalize=False,
                            title='Confusion matrix',
                            cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        from http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
        :param cm: (numpy matrix) confusion matrix
        :param classes: [str]
        :param normalize: (bool)
        :param title: (str)
        :param cmap: (matplotlib color map)
        """
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            
        plt.figure(figsize=(8, 8))   
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')