import numpy as np

from Poisoning.TriggerCreation.TriggerCreationMethod import TriggerCreationMethod

class PatchTrigger(TriggerCreationMethod):
    def __init__(self):
        self.is_clean_label = False

    def Fit(self, architecture, dataset, target_label):
        self.patch = np.zeros(dataset.ImgShape())

        #Patch is a yellow square in the bottom left corner.
        #(or a white one if the dataset is not colour.)
        self.patch[dataset.img_size-3:dataset.img_size, dataset.img_size-3:dataset.img_size] = np.array([1.0, 1.0, 0.0]) if dataset.isColour else 1.0

    def Apply(self, image, label):
        return np.clip(image + self.patch, 0, 1)
    
    def ApplyToAll(self, images, labels):
        return np.clip(images + self.patch, 0, 1)