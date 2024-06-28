from Poisoning.TriggerCreation.TriggerCreationMethod import TriggerCreationMethod

import numpy as np
from PIL import Image
import tensorflow as tf

class BlendTrigger(TriggerCreationMethod):
    #Image given should match the dataset images in shape.
    def __init__(self, image_path, alpha):
        self.is_clean_label = False
        
        image = Image.open(image_path)
        self.trigger = np.array(image) / 255.0
        if len(self.trigger.shape) == 2:
            self.trigger = np.expand_dims(self.trigger, axis=2)
        if self.trigger.shape[-1] == 4:
            self.trigger = self.trigger[:, :, 0:3] #Ignore alpha channel.
        self.alpha = alpha

    def Fit(self, architecture, dataset, target_label):
        if not dataset.isColour:
            self.trigger = self.trigger.mean(axis=2)
            self.trigger = tf.expand_dims(self.trigger, axis=2).numpy()
        if dataset.ImgShape() != self.trigger.shape:
            raise ValueError(f"The trigger image must match the shape of the dataset images. {self.trigger.shape} != {dataset.ImgShape()}")

    def Apply(self, image, label):
        return (1 - self.alpha) * image + self.alpha * self.trigger
    
    def ApplyToAll(self, images, labels):
        return (1 - self.alpha) * images + self.alpha * self.trigger