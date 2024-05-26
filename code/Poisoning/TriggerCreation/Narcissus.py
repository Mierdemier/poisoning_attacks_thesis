from Poisoning.TriggerCreation.TriggerCreationMethod import TriggerCreationMethod

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

loss_object = tf.keras.losses.CategoricalCrossentropy(reduction="sum")

class Narcissus(TriggerCreationMethod):
    def __init__(self, num_iterations, step_size, strength):
        self.is_clean_label = True
        self.num_iterations = num_iterations
        self.step_size = step_size
        self.strength = strength

        assert(num_iterations > 0)
        assert(step_size > 0)
        assert(strength > 0)

    @tf.function
    def LossGradient(self, surrogate, target_data, target_labels):
        #Get the gradient of the loss w.r.t. the trigger.
        with tf.GradientTape() as tape:
            tape.watch(target_data)
            loss = loss_object(target_labels, surrogate(target_data))
        gradient = tape.gradient(loss, target_data)

        #Return the average gradient
        return tf.reduce_mean(gradient, axis=0), loss / len(target_data)

    def Fit(self, architecture, dataset, target_label):
        #Acquire a surrogate.
        surrogate = architecture.CreateAndTrain(dataset.train_data, dataset.train_labels, is_clean_surrogate=True)
        target_data = dataset.train_data[self.ValidTargetMask(dataset.train_labels, target_label)]
        target_labels = np.tile(target_label, (len(target_data), 1))

        #Iteratively create the trigger.
        self.trigger = np.zeros(dataset.ImgShape())
        for i in range(self.num_iterations):
            gradient, loss = self.LossGradient(surrogate, self.ApplyToAll(target_data, target_labels), target_labels)
            
            self.trigger -= self.step_size * gradient.numpy()
            self.trigger = np.clip(self.trigger, -self.strength, self.strength)

            #For testing purposes: show how much you've moved the image towards the target class.
            if i % 15 == 0:
                print(f"Iteration {i}, loss {loss.numpy()}")

    def Apply(self, image, label):
        return np.clip(image + self.trigger, 0, 1)
    def ApplyToAll(self, images, labels):
        return np.clip(images + self.trigger, 0, 1)
    
    #Trigger is amplified at test time.
    def ApplyTesttime(self, image, label):
        return np.clip(image + 3 * self.trigger, 0, 1)
    def ApplyToAllTesttime(self, images, labels):
        return np.clip(images + 3 * self.trigger, 0, 1)