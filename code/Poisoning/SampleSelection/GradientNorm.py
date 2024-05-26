from Poisoning.SampleSelection.PoisoningMethod import PoisoningMethod
import numpy as np
import tensorflow as tf

def top_k_indices(arr, k):
    return np.argpartition(arr, -k)[-k:]

def bottom_k_indices(arr, k):
    return np.argpartition(arr, k)[:k]

loss_object = tf.keras.losses.CategoricalCrossentropy()

class GradientNorm(PoisoningMethod):
    def __init__(self):
        pass

    @tf.function
    def TFGradient(self, surrogate, image, label):
        with tf.GradientTape() as tape:
            prediction = surrogate(image)
            loss = loss_object(label, prediction)
        return tape.gradient(loss, surrogate.trainable_variables)
    
    def GradientNorms(self, surrogate, images, labels):
        norms = []
        for i in range(len(images)):
            img_tensor = tf.convert_to_tensor( np.array([images[i]]) )
            label_tensor = tf.convert_to_tensor( np.array([labels[i]]) )

            gradient = self.TFGradient(surrogate, img_tensor, label_tensor)
            gradient = np.concatenate([g.numpy().flatten() for g in gradient], axis=0)

            norms.append(np.linalg.norm(gradient))

        return np.array(norms)

    #Select images with the biggest gradient.
    def Poison(self, architecture, images, labels, trigger_creation_method, fraction_poisoned, target_label):
        surrogate = architecture.CreateAndTrain(images, labels, is_clean_surrogate=True)
        gradient_norms = self.GradientNorms(surrogate, images, labels)

        poisoned_indices = top_k_indices(gradient_norms, int(fraction_poisoned*len(images)))
        return trigger_creation_method.WithPoison(images, labels, poisoned_indices, target_label)