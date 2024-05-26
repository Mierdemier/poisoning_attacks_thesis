import numpy as np

#Use this as a template for creating new trigger creation methods.
class TriggerCreationMethod:
    def __init__(self):
        #Create any consistent trigger here.
        #e.g. for a patch trigger you would define the patch here.
        self.is_clean_label = False #Set this to true if you are a clean-label attack.

    def Fit(self, architecture, dataset, target_label):
        #Optional method to fit the trigger to a specific problem domain.
        pass

    def Apply(self, image, label):
        #Return this image with the trigger applied.
        #Be careful not to modify the original image.
        pass
    
    #Apply the trigger to a batch of images.
    #Does not modify the original images.
    def ApplyToIndices(self, images, labels, indices):
        return self.ApplyToAll(images[indices], labels[indices])
    
    #Override this if you can batch-apply the trigger to all images at once.
    def ApplyToAll(self, images, labels):
        new_images = []
        for image, label in zip(images, labels):
            new_images.append(self.Apply(image, label))
        return np.array(new_images)

    #Add the trigger to a batch of images, and add them to the original list.
    def WithPoison(self, images, labels, indices, target_label):
        new_images = np.concatenate((images, self.ApplyToIndices(images, labels, indices)), axis=0)
        new_labels = np.concatenate((labels, np.tile(target_label, (len(indices), 1))), axis=0)
        return new_images, new_labels
    
    #Mask that indicates which images this trigger can be added onto.
    def ValidTargetMask(self, labels, target_label):
        if self.is_clean_label:
            return labels.argmax(axis=1) == target_label.argmax()
        else:
            return labels.argmax(axis=1) != target_label.argmax()
        
    
    #Test time stuff. Override this if your trigger behaves differently at test time.
    def ApplyTesttime(self, image, label):
        return self.Apply(image, label)
    
    def ApplyToIndicesTesttime(self, images, labels, indices):
        return self.ApplyToAllTesttime(images[indices], labels[indices])
    
    def ApplyToAllTesttime(self, images, labels):
        new_images = []
        for image, label in zip(images, labels):
            new_images.append(self.ApplyTesttime(image, label))
        return np.array(new_images)