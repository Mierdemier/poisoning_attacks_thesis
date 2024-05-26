from DataManagement.Dataset import Dataset

#All triggers.
from Poisoning.TriggerCreation.TriggerCreationMethod import TriggerCreationMethod
from Poisoning.TriggerCreation.PatchTrigger import PatchTrigger
from Poisoning.TriggerCreation.BlendTrigger import BlendTrigger
from Poisoning.TriggerCreation.WarpTrigger import WarpTrigger
from Poisoning.TriggerCreation.StaticPerturbation import StaticPerturbation
from Poisoning.TriggerCreation.DynamicPerturbation import DynamicPerturbation
from Poisoning.TriggerCreation.Narcissus import Narcissus

#All poisoning methods.
from Poisoning.SampleSelection.PoisoningMethod import PoisoningMethod
from Poisoning.SampleSelection.RandomSampling import RandomSampling
from Poisoning.SampleSelection.FilterAndUpdate import FilterAndUpdate
from Poisoning.SampleSelection.ProxyFreeSelection import ProxyFreeSelection
from Poisoning.SampleSelection.GradientNorm import GradientNorm

import NeuralNetworks
import ComputationalCost

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow.keras
import timeit

def Find(arr, condition):
    for i, x in enumerate(arr):
        if condition(x):
            return i
    return -1

#Runs a test of a neural network architecture on a dataset w/ clean data.
#Returns the accuracy measured on a held out test set.
def TestClean(dataset, architecture):
    neuralnet = architecture.CreateAndTrain(dataset.train_data, dataset.train_labels)
    _, test_acc = architecture.Evaluate(neuralnet,
                                        dataset.test_data, 
                                        dataset.test_labels)
    print(f"Accuracy on clean data is {test_acc}")
    return test_acc

#Runs a test of a neural network architecuture on a dataset poisoned using the specified techniques.
#Returns a tuple of (clean accuracy, attack success rate) of the resulting poisoned network.
#Both are measured on a held out test set.
def TestAttack(dataset, architecture, trigger, poisoning_method, poisoned_fraction):
    target_label = np.zeros(dataset.NumLabels()); target_label[0] = 1

    trigger.Fit(architecture, dataset, target_label)
    poisoned_data, poisoned_labels = poisoning_method.Poison(architecture, 
                                                        dataset.train_data, 
                                                        dataset.train_labels, 
                                                        trigger, poisoned_fraction, target_label)
    neuralnet = architecture.CreateAndTrain(poisoned_data, poisoned_labels)

    _, test_acc = architecture.Evaluate(neuralnet, dataset.test_data, dataset.test_labels)
    print(f"Accuracy on clean data is {test_acc}")

    valid_target = dataset.test_labels.argmax(axis=1) != target_label.argmax()
    trigger_examples = trigger.ApplyToAllTesttime(dataset.test_data[valid_target], dataset.test_labels[valid_target])
    target_labels = np.tile(target_label, (len(trigger_examples), 1))
    _, trigger_success_rate = architecture.Evaluate(neuralnet,
                                                    trigger_examples, 
                                                    target_labels)
    print(f"Success rate of attack is {trigger_success_rate}")

    return test_acc, trigger_success_rate

#Searches a specific digit behind the comma. Returns the lowest digit that works.
def SearchDigit(tested_poisoning_fractions, accuracies, success_rates, precision, addition=0, num_tries=3):
    #Options. Change these to fit your current experiment.
    get_clean_dataset = Dataset.Cifar10
    dataset = get_clean_dataset()
    architecture = NeuralNetworks.Architecture.SimpleConvolutional(dataset.ImgShape(), dataset.NumLabels())
    
    trigger = PatchTrigger()
    #trigger = BlendTrigger("Blend_Trigger.png", 0.10)
    #trigger = WarpTrigger(4, 0.5)
    #trigger = StaticPerturbation(2, 0.04)
    #trigger = DynamicPerturbation(3, 0.04)
    #trigger = Narcissus(50, 1, 0.06)

    poisoning_method = RandomSampling()
    #poisoning_method = ProxyFreeSelection(10)
    #poisoning_method = FilterAndUpdate(10, 0.5)
    #poisoning_method = GradientNorm()

    upper_bound = 10
    for digit in range(1, upper_bound):
        poisoning_fraction = addition + digit * precision

        #Skip if we already tested this poisoning fraction.
        if any(abs(x - poisoning_fraction) < 0.00001 for x in tested_poisoning_fractions):
            continue

        print(f"Attempting poisoning fraction: {poisoning_fraction}")
        #Perform attack.
        avg_accuracy = 0; avg_success = 0
        for i in range(num_tries):
            tf.keras.backend.clear_session()
            architecture.ResetSurrogate()
            accuracy, success_rate = TestAttack(dataset, architecture, trigger, poisoning_method, poisoning_fraction)
            avg_accuracy += accuracy / num_tries; avg_success += success_rate / num_tries
            dataset = get_clean_dataset() #Reset dataset.

            if avg_success + (num_tries - 1 - i) / num_tries < 0.90:
                print("Breaking early.")
                avg_success *= num_tries / (i + 1)
                avg_accuracy *= num_tries / (i + 1)
                break #No need to continue if we can't reach 90% success rate.
        
        #Record results.
        print(f"Accuracy: {avg_accuracy}, success rate {avg_success}")
        tested_poisoning_fractions.append(poisoning_fraction)
        accuracies.append(avg_accuracy)
        success_rates.append(avg_success)

        #Search until we find a poisoning fraction that works.
        if avg_success >= 0.90:
            return digit
    return upper_bound

#Runs a single attack experiment to discover the necessary poisoned_fraction to make that attack work.
#The lists are by-reference, so you can access them after the function is done to find more information.
def Search(tested_poisoning_fractions, accuracies, success_rates, num_tries=3, multiplier=0.10):
    first_working_idx = Find(success_rates, lambda x: x >= 0.90)
    if first_working_idx != -1:
        print("Already found first figure.")
        first_figure = int(round(tested_poisoning_fractions[first_working_idx] / multiplier)) - 1
    else:
        first_figure = SearchDigit(tested_poisoning_fractions, accuracies, success_rates, 
                                   multiplier, num_tries=num_tries) - 1
    print(f"First figure is determined to be {first_figure}")

    second_figure = SearchDigit(tested_poisoning_fractions, accuracies, success_rates, 
                                multiplier*0.10, num_tries=num_tries, addition=first_figure*multiplier)
    print(f"Second figure is determined to be {second_figure}")
    return multiplier * first_figure + 0.10 * multiplier * second_figure

def PlotSearch(title, tested_poisoning_fractions, accuracies, success_rates):
    plt.scatter(tested_poisoning_fractions, accuracies, label="Accuracy on clean test data.")
    plt.scatter(tested_poisoning_fractions, success_rates, label="Trigger success rate")

    plt.xlabel("Poisoned fraction")
    plt.xlim(0, 1)
    plt.ylim(0, 1)

    plt.legend()
    plt.title(title)
    plt.show()

#Performs the reality check experiment and returns the average result.
def RealityCheck(get_clean_dataset, architecture, trigger, poisoning_method, poisoned_fraction):
    target_label = np.zeros(10); target_label[0] = 1

    avg_time = 0
    for _ in range(3):
        tf.keras.backend.clear_session()
        dataset = get_clean_dataset()
        architecture.ResetSurrogate()


        start = timeit.default_timer()
        #Note: Only the stuff the attacker does is included here! 
        #Training the final network is done by the victim.
        trigger.Fit(architecture, dataset, target_label)
        poisoned_data, poisoned_labels = poisoning_method.Poison(architecture, 
                                                        dataset.train_data, 
                                                        dataset.train_labels, 
                                                        trigger, poisoned_fraction, target_label)
        end = timeit.default_timer()


        avg_time += (end - start) / 3
    print(f"Average time is {avg_time}")
    return avg_time


def main():
    fractions = []
    accs =      []
    rates =     []

    try:
        frac = Search(fractions, accs, rates)
        print(f"Poisoning fraction is {frac}")
        print(f"Clean accuracy: {accs[-1]}")

        PlotSearch("Patch Trigger | Random Sampling", fractions, accs, rates)

    except:
        print("Error occurred. Printing results so you can continue where you left off.")
    finally:
        print(fractions)
        print(accs)
        print(rates)
        
        
if __name__ == "__main__":
    main()