#!/usr/bin/python2.7
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


if __name__ == '__main__':

    classifier = 

    # classifier.runWithContext()


    # ctx = Context(sys.argv)
    
    

