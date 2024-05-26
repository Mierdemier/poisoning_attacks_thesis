import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa #You might need to find a whl of this library online.

from Poisoning.TriggerCreation.TriggerCreationMethod import TriggerCreationMethod

def clamp(n, smallest, largest): return max(smallest, min(n, largest))

class WarpTrigger(TriggerCreationMethod):
    def __init__(self, k, strength):
        self.is_clean_label = False
        
        #Construct a random base for the warp field.
        self.warp_field = np.random.rand(k, k, 2) * 2 - 1
        self.warp_field /= np.mean(np.abs(self.warp_field))
        self.warp_field *= strength

    def Fit(self, architecture, dataset, target_label):
        #Resize the warp field to the image size.
        self.warp_field = tf.image.resize(self.warp_field, (dataset.img_size, dataset.img_size), method='bicubic')
        self.warp_field = self.warp_field.numpy()

        #Clip the warp field so it does not access pixels outside of image bounds.
        for i in range(dataset.img_size):
            for j in range(dataset.img_size):
                self.warp_field[i, j, 0] = clamp(self.warp_field[i, j, 0], -i, dataset.img_size - i - 1)
                self.warp_field[i, j, 1] = clamp(self.warp_field[i, j, 1], -j, dataset.img_size - j - 1)


    def Apply(self, image, label):
        image_batch = np.array([image])
        warp_field_batch = np.array([self.warp_field])
        poisoned_image = tfa.image.dense_image_warp(image_batch, warp_field_batch)[0]
        return poisoned_image.numpy()
    
    def ApplyToAll(self, images, labels):
        image_batch = np.array(images)
        warp_field_batch = np.array([self.warp_field for _ in images])
        poisoned_images = tfa.image.dense_image_warp(image_batch, warp_field_batch)
        return poisoned_images.numpy()