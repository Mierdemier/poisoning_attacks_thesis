import random
import numpy as np

from Poisoning.TriggerCreation.TriggerCreationMethod import TriggerCreationMethod

class StaticPerturbation(TriggerCreationMethod):
    def __init__(self, region_size, perturbation_strength):
        self.is_clean_label = False
        self.region_size = region_size
        self.perturbation_strength = perturbation_strength
        
    def Fit(self, architecture, dataset, target_label):
        img_size = dataset.img_size
        
        i = random.randint(0, self.region_size - 1)
        j = random.randint(0, self.region_size - 1)
        region = np.zeros((self.region_size, self.region_size, 3)) if dataset.isColour else np.zeros((self.region_size, self.region_size))
        region[i, j] = self.perturbation_strength
        
        self.perturbation = np.tile(region, (img_size // self.region_size + 1, img_size // self.region_size + 1, 1))[:img_size, :img_size]

    def Apply(self, image, label):
        return np.clip(image + self.perturbation, 0, 1)
    
    def ApplyToAll(self, images, labels):
        return np.clip(images + self.perturbation, 0, 1)