import numpy as np
from Poisoning.SampleSelection.PoisoningMethod import PoisoningMethod

class RandomSampling(PoisoningMethod):
    def __init__(self):
        pass

    def Poison(self, architecture, images, labels, trigger_creation_method, fraction_poisoned, target_label):
        #Randomly select the fraction of the dataset that should be poisoned.
        n = len(images)
        poisonable_indices = np.arange(n)[trigger_creation_method.ValidTargetMask(labels, target_label)]
        indices = np.random.choice(poisonable_indices, int(n*fraction_poisoned), replace=False)
        return trigger_creation_method.WithPoison(images, labels, indices, target_label)