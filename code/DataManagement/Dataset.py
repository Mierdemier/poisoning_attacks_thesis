import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import numpy as np
import numbers

class Dataset:
    def __init__(self, train_data, train_labels, test_data, test_labels, img_size, isColour, label_names=None):
        if len(train_data) != len(train_labels):
            raise ValueError("The number of training data points and labels must be the same.")
        if len(test_data) != len(test_labels):
            raise ValueError("The number of test data points and labels must be the same.")
        if label_names and (len(train_labels[0]) != len(label_names)):
            raise ValueError("THe labels must be one-hot encodings, and the label_names must be a list of names of the same length as an encoding.")


        self.train_data = train_data
        self.train_labels = train_labels
        self.test_data = test_data
        self.test_labels = test_labels
        self.img_size = img_size
        self.isColour = isColour
        self.label_names = label_names if label_names else [str(i) for i in range(len(train_labels[0]))]


    @staticmethod
    def Cifar10():
        (x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
        x_train, x_test = x_train / 255.0, x_test / 255.0
        y_train = tf.keras.utils.to_categorical(y_train, 10)
        y_test = tf.keras.utils.to_categorical(y_test, 10)

        labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        return Dataset(x_train, y_train, x_test, y_test, 32, True, labels)

    @staticmethod
    def Mnist():
        (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()

        #Pad to make it 32x32
        x_train = tf.pad(x_train, [[0, 0], [2,2], [2,2]]) / 255
        x_test = tf.pad(x_test, [[0, 0], [2,2], [2,2]]) / 255
        #Expand dimension to include colour axis.
        x_train = tf.expand_dims(x_train, axis=3).numpy()
        x_test = tf.expand_dims(x_test, axis=3).numpy()
        
        y_train = tf.keras.utils.to_categorical(y_train, 10)
        y_test = tf.keras.utils.to_categorical(y_test, 10)

        return Dataset(x_train, y_train, x_test, y_test, 32, False)

    def __len__(self):
        return len(self.train_data) + len(self.test_data)
    
    def __getitem__(self, index):
        if index < len(self.train_data):
            return self.train_data[index]
        else:
            return self.test_data[index - len(self.train_data)]
        
    def __iter__(self):
        for i in range(len(self)):
            yield self[i]
        
    def LabelName(self, label):
        if isinstance(label, numbers.Integral):
            return self.label_names[label]
        return self.label_names[np.argmax(label)]
    
    def NumLabels(self):
        return len(self.label_names)
    
    def ImgShape(self):
        return (self.img_size, self.img_size, 3) if self.isColour else (self.img_size, self.img_size, 1)
    
    def GetTrainImage(self, index):
        return self.train_data[index]
    def GetTestImage(self, index):
        return self.test_data[index]
    
    def GetTrainLabel(self, index):
        return self.train_labels[index]
    def GetTestLabel(self, index):
        return self.test_labels[index]
    
    def GetAllTrainImages(self):
        return self.train_data
    def GetAllTestImages(self): 
        return self.test_data
    
    def GetAllTrainLabels(self):    
        return self.train_labels
    def GetAllTestLabels(self):
        return self.test_labels