#This is a template to base your own poisoning method on.
#Trigger creation method is a parameter, so you should focus on the sample selcetion method.
class PoisoningMethod:
    #Set global parameters that are independent of dataset and trigger creation method here.
    def __init__(self):
        pass

    #Poison whichever images you want from the dataset and return a tuple (all_images, all_labels) (including both poisoned and clean ones).
    #You may modify the images and labels in place if you desire.
    def Poison(self, architecture, images, labels, trigger_creation_method, fraction_poisoned, target_label):
        pass
