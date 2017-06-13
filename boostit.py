#!/usr/bin/python3
# -*- coding: utf-8 -*-
# coding: utf-8
import sys
import unicodedata
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
    ℕ_data_points = 200
    # ⇒, Σ
    w⃗ = (1/cardinality_D for x in range(len(eval("{0} * [None]".format(cardinality_D)))))
    # 𝒜: Learning Algorithm
    # 𝓐 =
    # 𝓓 = data
    # "λ
    # 𝜀
    # 𝛼
    # 𝛵: ensemble_size
    # algFunc = None
    # model = weakModel(algFunc, weight_vec)
    f = lambda x: sin(x) if 0 <= x <= 2*pi else 0

    for word in w⃗:
        print (word)
    # classifier.runWithContext()


    # ctx = Context(sys.argv)
