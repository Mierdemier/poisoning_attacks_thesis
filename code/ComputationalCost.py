import numpy as np
import tensorflow as tf

from Poisoning.TriggerCreation.WarpTrigger import WarpTrigger

import timeit
import random

def Measure(operation, dataset):
    start = timeit.default_timer()
    _ = operation(dataset.train_data)
    end = timeit.default_timer()
    return end - start


def Addition(dataset):
    to_add = np.random.random_sample(dataset.ImgShape())
    return Measure(lambda x: x + to_add, dataset)

def Subtraction(dataset):
    to_subtract = np.random.random_sample(dataset.ImgShape())
    return Measure(lambda x: x - to_subtract, dataset)

def Multiplication(dataset):
    #Note: I tested both options separately at one point, and found that the difference does not matter.
    factor = random.choice([random.randint(2, 5), random.random()])
    return Measure(lambda x: x * factor, dataset)

def Clip(dataset):
    s = random.random()
    #Note: I tested both options separately at one point, and found that the difference does not matter.
    lower_bound, upper_bound = random.choice([(0, 1), (-s, s)])
    return Measure(lambda x: np.clip(x, lower_bound, upper_bound), dataset)

def Warp(dataset):
    trigger = WarpTrigger(4, 0.5)
    trigger.Fit(None, dataset, None)
    return Measure(lambda x: trigger.ApplyToAll(x, None), dataset)

def Norm(dataset):
    return Measure(lambda x: np.linalg.norm(x, axis=(1,2)), dataset)
    


#Neural network stuff

def Forward(neuralnet, dataset):
    return Measure(neuralnet.predict, dataset)
    
def Training(architecture, dataset):
    neuralnet = architecture.Create()
    start = timeit.default_timer()
    neuralnet.fit(dataset.train_data, dataset.train_labels, epochs=architecture.num_epochs)
    end = timeit.default_timer()
    return end - start


#Attack optimisation
loss_object = tf.keras.losses.CategoricalCrossentropy()
@tf.function
def _GetGradient(neuralnet, data, labels):
    with tf.GradientTape() as tape:
        tape.watch(data)
        loss = loss_object(labels, neuralnet(data))
    gradient = tape.gradient(loss, data)
    return tf.reduce_mean(gradient, axis=0)

def FindImageGradient(neuralnet, dataset):
    start = timeit.default_timer()
    _ = _GetGradient(neuralnet, dataset.train_data, dataset.train_labels).numpy()
    end = timeit.default_timer()
    return end - start


@tf.function
def _GetMultipleGradients(neuralnet, data, labels):
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(data)
        prediction = neuralnet(data)
        a = prediction[0, 0]
        b = prediction[0, 1]
    gradientA = tape.gradient(a, data)
    gradientB = tape.gradient(b, data)
    return gradientA, gradientB

def FindTwoGradients(neuralnet, dataset):
    start = timeit.default_timer()
    (x, y) = _GetMultipleGradients(neuralnet, dataset.train_data, dataset.train_labels)
    x = x.numpy(); y = y.numpy()
    end = timeit.default_timer()
    return end - start

@tf.function  
def _GetParameterGradient(neuralnet, data, labels):
    with tf.GradientTape() as tape:
        tape.watch(neuralnet.trainable_variables)
        loss = loss_object(labels, neuralnet(data))
    gradient = tape.gradient(loss, neuralnet.trainable_variables)
    return gradient

def FindParameterGradient(neuralnet, dataset):
    start = timeit.default_timer()
    x = _GetParameterGradient(neuralnet, dataset.train_data, dataset.train_labels)
    #x = np.concatenate([g.numpy().flatten() for g in x], axis=0) This does not make a difference.
    end = timeit.default_timer()
    return end - start
