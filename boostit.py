#!/usr/bin/python3
import sys
from collections import namedtuple

TrainingFile = namedtuple("TrainingFile",["pos","neg"])
TestingFile = namedtuple("TestingFile", ["pos", "neg"])


class Context:
    def __init__(self, usage):
        if len(usage) != 6:
            print("Error: usage is <ensemble size> <pos-train-file> "
                  "<neg-train-file> <pos-test-file> <neg-test-file>\n\nGiven input:\t{0}".format(usage))
            sys.exit(1)

        self.ensemble_size = usage[1]
        self.training_files = TrainingFile(usage[2], usage[3])
        self.testing_files = TestingFile(usage[4], usage[5])

# class Pseudo:
    # @args: Data type, ensemble size T, learning algorithm A
    # @return: weighted ensemble of models T
    # def __init__(self, data, ensemble_size, learning_algorithm):
        # vec = [float(1) / data.size] * data.size  # Non-lazy eval
        # weight_vec = (1/2 for x in range(data.size))



if __name__ == '__main__':
    # weight_vec = [None] * 100
    cardinality_D = 100
    weight_vec = (1/cardinality_D for x in range(len(eval("{0} * [None]".format(cardinality_D)))))
    # algFunc = None
    # model = weakModel(algFunc, weight_vec)
    for word in classifier:
        print (word)


    # classifier.runWithContext()


    # ctx = Context(sys.argv)
