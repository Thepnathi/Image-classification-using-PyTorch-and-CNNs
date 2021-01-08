# Introduction

Image classification using PyTorch and Convolutional Neural Network.

## How to run the program

How to run the different steps of the CNN model (and the time it takes).
Paste the python commands into console/terminal.

### Step 1 - Train the network

* Train the CNN model using 350 images of 5 classes
* Save the trained model (we include pre-trained models so the file
* will show the training history of those models.)
```
python train_network.py
```

### Step 2 - Test the network on the test data and report the results

* Load all the trained models
* For each model, calculate the overall accuracy of the test and training images
```
python dataset_accuracy.py
```
* Load all the trained models
* For each model, calculate the accuracy of each class
* Compute the confusion matrix
* Display the confusion matrix with Matplotlib
```
python class_accuracy.py
```