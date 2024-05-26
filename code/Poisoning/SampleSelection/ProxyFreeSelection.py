import numpy as np
from scipy.spatial.distance import cosine

from Poisoning.SampleSelection.PoisoningMethod import PoisoningMethod
import NeuralNetworks


def top_k_indices(arr, k):
    return np.argpartition(arr, -k)[-k:]

def bottom_k_indices(arr, k):
    return np.argpartition(arr, k)[:k]

class ProxyFreeSelection(PoisoningMethod):
    def __init__(self, diversity):
        if diversity < 1:
            raise ValueError("Diversity must be at least 1.")

        self.diversity = diversity

    def TrainFeatureExtractor(self, architecture, images, labels):
        self.feature_extractor = architecture.CreateAndTrain(images, labels, is_clean_surrogate=True)
        self.feature_extractor = NeuralNetworks.FirstLayersOf(self.feature_extractor, architecture.input_shape)

    def Poison(self, architecture, images, labels, trigger_creation_method, fraction_poisoned, target_label):
        self.TrainFeatureExtractor(architecture, images, labels)
        
        clean_images = images[trigger_creation_method.ValidTargetMask(labels, target_label)]
        clean_labels = labels[trigger_creation_method.ValidTargetMask(labels, target_label)]
        poisoned_images = trigger_creation_method.ApplyToAll(clean_images, clean_labels)
        n_total = len(images)
        n_poisonable = len(clean_images)

        distances = np.zeros(n_poisonable)
        clean_features = architecture.Predict(self.feature_extractor, clean_images)
        poisoned_features = architecture.Predict(self.feature_extractor, poisoned_images)
        for i in range(n_poisonable):
            distances[i] = cosine(clean_features[i], poisoned_features[i])

        coarse_length = int(n_total * fraction_poisoned * self.diversity)
        coarse_length = min(coarse_length, n_poisonable // 2)
        #Note: the bottom k *least* distant are the *most* similar.
        coarse_indices = bottom_k_indices(distances, coarse_length)

        final_indices = np.random.choice(coarse_indices, int(n_total * fraction_poisoned), replace=False)

        return np.concatenate((images, poisoned_images[final_indices]), axis=0), np.concatenate((labels, np.tile(target_label, (len(final_indices), 1)) ), axis=0)        