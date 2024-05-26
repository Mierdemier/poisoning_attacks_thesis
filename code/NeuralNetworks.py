import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import numpy as np


#Defines a plan for creating and training a neural network.
#This will help us to keep track of surrogate networks, etc.
class Architecture:
    def __init__(self, input_shape, num_labels, model_type, num_epochs, preprocess = None):
        self.clean_surrogate = None
        self.input_shape = input_shape
        if len(self.input_shape) == 2:
            self.input_shape += (1,)

        self.num_labels = num_labels
        self.model_type = model_type #A function that takes input_shape and num_labels and returns a model.
        self.num_epochs = num_epochs
        self.preprocess = preprocess #A function that takes a list of images and returns a list of images.

    def ResetSurrogate(self):
        self.clean_surrogate = None

    def Create(self):
        return self.model_type(self.input_shape, self.num_labels)
    
    def Train(self, model, train_data, train_labels, is_clean_surrogate=False):
        if is_clean_surrogate and self.clean_surrogate is not None:
            model.set_weights(self.clean_surrogate.get_weights())
            return
            
        if self.preprocess:
            train_data = self.preprocess(train_data)
        model.fit(train_data, train_labels, epochs=self.num_epochs)

        if is_clean_surrogate:
            self.clean_surrogate = model

    def TrainWithValidation(self, model, train_data, train_labels, validation_data, validation_labels):
        if self.preprocess:
            train_data = self.preprocess(train_data)
            validation_data = self.preprocess(validation_data)
        return model.fit(train_data, train_labels, epochs=self.num_epochs, validation_data=(validation_data, validation_labels))

    def CreateAndTrain(self, train_data, train_labels, is_clean_surrogate=False):
        model = self.Create()
        self.Train(model, train_data, train_labels, is_clean_surrogate)
        return model
    
    def Evaluate(self, model, test_data, test_labels):
        if self.preprocess:
            test_data = self.preprocess(test_data)
        return model.evaluate(test_data, test_labels, verbose=2)
    
    def Predict(self, model, images):
        if self.preprocess:
            images = self.preprocess(images)
        return model.predict(images)
    
    @staticmethod
    def SimpleConvolutional(input_shape, num_labels):
        return Architecture(input_shape, num_labels, SimpleConvolutional, 10)
    
    @staticmethod
    def LeNet(input_shape, num_labels):
        return Architecture(input_shape, num_labels, LeNet, 40)

    @staticmethod
    def MobileNet(input_shape, num_labels):
        if len(input_shape) < 3 or input_shape[2] != 3:
            input_shape = input_shape[:2] + (3,)
        if input_shape[0] < 32 or input_shape[1] < 32:
            input_shape = (32, 32, 3)
        return Architecture(input_shape, num_labels, MobileNet, 25, MobileNetPreprocess)

def SimpleConvolutional(input_shape, num_labels):
    model = models.Sequential()

    #input layer
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    
    #Convolutional part.
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())

    #Dense part.
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(num_labels, activation='softmax')) #output layer

    model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

    return model


def LeNet(input_shape, num_labels):
    model = models.Sequential()
    #input layer
    model.add(layers.Conv2D(6, (5, 5), activation='tanh', input_shape=input_shape, padding='same'))

    #Convolutional part.
    model.add(layers.AveragePooling2D(2))
    model.add(layers.Activation('sigmoid'))
    model.add(layers.Conv2D(16, 5, activation='tanh', padding='valid'))
    model.add(layers.AveragePooling2D(2))
    model.add(layers.Activation('sigmoid'))
    model.add(layers.Conv2D(120, 5, activation='tanh', padding='valid'))
    model.add(layers.Flatten())

    #Dense part.
    model.add(layers.Dense(84, activation='tanh'))
    model.add(layers.Dense(num_labels, activation='softmax')) #output layer

    model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def MobileNetPreprocess(images):
    #MobileNet expects images in the range [-1, 1]
    return images * 2.0 - 1.0


def MobileNet(input_shape = (32, 32, 3), num_labels = 10):
    model = tf.keras.applications.MobileNetV3Small(input_shape=input_shape, 
                                                  classes=num_labels, 
                                                  weights=None, 
                                                  include_preprocessing=False)
    model.compile('rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def FirstLayersOf(network, input_shape):
    input_layer = tf.keras.Input(shape=input_shape)
    x = input_layer

    # Add the first n layers from the old model to the new model
    for layer in network.layers[:-1]:
        x = layer(x)

    # Create the new model
    new_model = tf.keras.Model(inputs=input_layer, outputs=x)  
    return new_model 