import numpy as np
import tensorflow as tf

from Poisoning.SampleSelection.PoisoningMethod import PoisoningMethod

def top_k_indices(arr, k):
    return np.argpartition(arr, -k)[-k:]

def bottom_k_indices(arr, k):
    return np.argpartition(arr, k)[:k]

class FilterAndUpdate(PoisoningMethod):
    def __init__(self, num_iterations, filtering_ratio):
        self.num_iterations = num_iterations
        self.filtering_ratio = filtering_ratio
        self.num_forgettings = None

    #Trains a surrogate model on the given datapoints, and returns it.
    #The number of forgettings for each sample is recorded in self.num_forgettings.
    def TrainSurrogateModel(self, architecture, images, labels, poisoned_indices, trigger, target_label):
        if architecture.num_epochs >= 255:
            raise ValueError("More than 255 epochs could cause the forgettings to overflow!")

        num_unpoisoned = len(images)
        images, labels = trigger.WithPoison(images, labels, poisoned_indices, target_label)
        if architecture.preprocess:
            images = architecture.preprocess(images)

        tf.keras.backend.clear_session()
        #While training the model we will also remember how often each sample is forgotten.
        #Note that the trigger will add the poisoned samples to the end of the dataset in the same order as poisoned_indices.
        self.num_forgettings = np.zeros(len(poisoned_indices))

        model = architecture.Create()
        prev_correct_predictions = np.full(len(poisoned_indices), False)
        for _ in range(architecture.num_epochs):
            model.fit(images, labels, epochs=1, verbose=0)

            correct_predictions = model.predict(images[num_unpoisoned:]).argmax(axis=1) == target_label.argmax()
            forgettings = prev_correct_predictions & (~correct_predictions)
            self.num_forgettings += forgettings

            prev_correct_predictions = correct_predictions
        #print(f"The average number of forgettings among the poisoned samples was {np.mean(self.num_forgettings)}.")
        return model

    def Poison(self, architecture, images, labels, trigger_creation_method, fraction_poisoned, target_label):
        #Sample some random points to poison.
        n = len(images)
        poisonable_indices = np.arange(n)[trigger_creation_method.ValidTargetMask(labels, target_label)]

        #poisoned indices are the indices of the poisoned samples.
        poisoned_indices = np.random.choice(poisonable_indices, int(n*fraction_poisoned), replace=False)

        for i in range(self.num_iterations):
            print(f"Sampling iteration {i+1}/{self.num_iterations}.")

            #Train an infected model, while recording the forgetability of the samples.
            self.TrainSurrogateModel(architecture, images, labels, poisoned_indices, trigger_creation_method, target_label)

            #Remove the k least forgettable samples.
            #This is the same as retaining the n_samples - k most forgettable samples.
            k = int(len(poisoned_indices)*self.filtering_ratio)

            poisoned_indices = poisoned_indices[top_k_indices(self.num_forgettings, len(poisoned_indices)-k)]

            #Put in equally many random new samples.
            #Note that it is technically possible for the same sample to be selected again.
            #The original paper makes no effort to stop this, so I won't either.
            new_indices = np.random.choice(poisonable_indices, k, replace=False)
            poisoned_indices = np.concatenate((poisoned_indices, new_indices), axis=0)
            

        #Definitively poison the final images that were selected.
        return trigger_creation_method.WithPoison(images, labels, poisoned_indices, target_label)