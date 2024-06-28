import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from Poisoning.TriggerCreation.TriggerCreationMethod import TriggerCreationMethod

class DynamicPerturbation(TriggerCreationMethod):
    def __init__(self, num_iterations, strength):
        self.is_clean_label = False
        self.num_iterations = num_iterations
        self.strength = strength

    #Creates a surrogate model.
    def TrainSurrogate(self, architecture, dataset):
        model = architecture.CreateAndTrain(dataset.train_data, dataset.train_labels, is_clean_surrogate=True)

        # This method requires a surrogate that does not softmax (for better gradients)
        config = model.layers[-1].get_config()
        weights = [x.numpy() for x in model.layers[-1].weights]

        config['activation'] = tf.keras.activations.linear
        config['name'] = 'logits'

        new_layer = tf.keras.layers.Dense(**config)(model.layers[-2].output)
        new_model = tf.keras.Model(inputs=[model.input], outputs=[new_layer])
        new_model.layers[-1].set_weights(weights)

        return new_model
    
    @tf.function
    def GetGradients(self, test_img, surrogate, source_class, target_class):
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(test_img)
            output = surrogate(test_img)
            target_output = output[0, target_class]
            source_output = output[0, source_class]
        target_gradient = tape.gradient(target_output, test_img)[0]
        source_gradient = tape.gradient(source_output, test_img)[0]

        return target_gradient, source_gradient, output[0]

    #Performs one iteration of the deepfool algorithm, updating the perturbation to point towards the target class.
    #This currently does not work with neural networks that require pre-processing.
    def IterateDeepfool(self, img, architecture, surrogate, source_label, target_label):
        source_class = source_label.argmax(); target_class = target_label.argmax()

        #Test the current best perturbation on this image and record the gradients in the surrogate model.
        #We want to compute the gradient of output[target_class] d perturbation_test
        #                              and of output[source_class] d perturbation_test
        #'surrogate' is a neural network in the form of a tensorflow keras model.
        perturbation_test = tf.convert_to_tensor( np.array([self.Apply(img, source_label)]) )
        target_gradient, source_gradient, output = self.GetGradients(perturbation_test, surrogate, source_class, target_class)
        target_gradient = target_gradient.numpy(); source_gradient = source_gradient.numpy(); output = output.numpy()

        if output.argmax() == target_class:
            return np.zeros(img.shape), 1
        target_output = output[target_class]
        source_output = output[source_class]

        #Find a way to get to the decision boundary with the target class.
        perturbation_dir = target_gradient - source_gradient
        diff = float(abs(target_output - source_output))

        update_size = diff / np.linalg.norm(perturbation_dir.flatten())

        perturbation_update = (update_size+1e-4) * perturbation_dir / np.linalg.norm(perturbation_dir)
        return perturbation_update, 0

    # Normalises a perturbation according to the Lmax norm.
    def Normalise(self, perturbation):
        return np.clip(perturbation, -self.strength, self.strength)

    # Constructs a dynamic perturbation for the source class.
    # The result is stored in self.perturbations.
    def AdaptPerturbations(self, architecture, surrogate, train_data, train_labels, target_label):
        for _ in range(self.num_iterations):
            n_images = 0
            n_fooled = 0
            for (img, label) in zip(train_data, train_labels):
                source_class = label.argmax()

                #Skip if it is already in the target class.
                if target_label.argmax() == source_class:
                    continue

                #Update the perturbation.
                change, fooled = self.IterateDeepfool(img, architecture, surrogate, label, target_label)
                self.perturbations[source_class] = self.Normalise(self.perturbations[source_class] + change)
                n_fooled += fooled
                n_images += 1
            print(f"Fooled {n_fooled / n_images} fraction of images.")


    def Fit(self, architecture, dataset, target_label):
        surrogate = self.TrainSurrogate(architecture, dataset)
        
        #Normally, dynamic perturbation targets a specific source class.
        #In our case, we want to target every class.
        #So, we will create a perturbation for every class.
        self.perturbations = [np.zeros(dataset.ImgShape()) for _ in range(dataset.NumLabels())]
        self.AdaptPerturbations(architecture, surrogate, dataset.train_data, dataset.train_labels, target_label)

    def Apply(self, image, label):
        return np.clip(image + self.perturbations[label.argmax()] * 1.02, 0, 1)