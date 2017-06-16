#!/usr/bin/python3
# -*- coding: utf-8 -*-
# coding: utf-8
import sys
import unicodedata
from collections import namedtuple
import numpy as np
from math import log
from cmath import log

import re

TrainingFile = namedtuple("TrainingFile",["pos","neg"])
TestingFile = namedtuple("TestingFile", ["pos", "neg"])
# 𝓓 = namedtuple("𝓓", ["pos", "neg"])

# 𝓓Training

# Python 3.0+
# from abc import ABCMeta, abstractmethod
# class Abstract(metaclass=ABCMeta):
#     @abstractmethod
#     def foo(self):
#         pass


class EnsembleMethod(object):

    class Context:
        def __init__(self, usage):
            if len(usage) != 6:
                print("Error: usage is <ensemble size> <pos-train-file> "
                      "<neg-train-file> <pos-test-file> <neg-test-file>\n\nGiven input:\t{0}".format(usage))
                sys.exit(1)

            self.ensemble_size = usage[1]  # ensemble_size
            self.training_files = TrainingFile(usage[2], usage[3])
            self.testing_files = TestingFile(usage[4], usage[5])
            self.pos_training_file = usage[2]
            self.neg_training_file = usage[3]
            self.pos_testing_file = usage[4]
            self.neg_testing_file = usage[5]
            self.output_filename = self.interpret_output_filename(self.pos_training_file)

            # Get predictors, also known as x_vector or feature_vector
            self.pos_training_predictors, self.pos_num_training_samples, self.pos_training_dimensionality = self.load_data(
                self.pos_training_file)
            self.neg_training_predictors, self.neg_num_training_samples, self.neg_training_dimensionality = self.load_data(
                self.neg_training_file)
            training_predictors = np.concatenate((self.pos_training_predictors, self.neg_training_predictors), axis=0)

            self.pos_testing_predictors, self.pos_num_testing_samples, self.pos_testing_dimensionality = self.load_data(self.pos_testing_file)
            self.neg_testing_predictors, self.neg_num_testing_samples, self.neg_testing_dimensionality = self.load_data(self.neg_testing_file)
            self.testing_predictors = np.concatenate((self.pos_testing_predictors, self.neg_testing_predictors), axis=0)

            # Get targets, also known as Y, actual, or labels
            self.pos_training_targets = np.ones(self.pos_num_training_samples, dtype=int)
            self.neg_training_targets = np.full(self.neg_num_training_samples, fill_value=-1, dtype=int)
            # y_training_targets = np.zeros(pos_num_training_samples + neg_num_training_samples, dtype=int)
            self.y_training_targets = np.concatenate((self.pos_training_targets, self.neg_training_targets))

            self.pos_testing_targets = np.ones(self.pos_num_testing_samples, dtype=int)
            self.neg_testing_targets = np.full(self.neg_num_testing_samples, fill_value=-1, dtype=int)
            self.y_testing_targets = np.concatenate((self.pos_testing_targets, self.neg_testing_targets))
            self.ground_truth_targets = map(lambda y: 0 if y == -1 else 1, self.y_testing_targets)

            self.num_training = len(self.y_training_targets)
            self.num_testing = len(self.y_testing_targets)

            # clf_model = PerceptronModel(training_predictors, y_training_targets, num_training,
            #                             kernelmodel=('rbf', sigma))
            # similarities = clf_model.parameterize_RBF(training_predictors, training_predictors)
            # clf_model.converge_training_weights(clf_model.alphas)
            # predicted = clf_model.predict(testing_predictors, y_testing_targets)
            #
            # alphas_list = ' '.join('{0}'.format(v, i) for i, v in enumerate(clf_model.alphas))
            # print
            # 'Alphas:\t' + alphas_list

        def interpret_output_filename(self, file_with_digits):
            try:
                file_no = int(re.match('.+([0-9]+)[^0-9]*$', file_with_digits).group(1))
                outputfile = 'output{0}.txt'.format(file_no)
            except (ValueError, AttributeError) as e:
                outputfile = 'output.txt'
            return outputfile


        def load_data(self, training_file):
            '''
            Transform a data file into a np ndarray with any N features.
            The first column is ignored because the record number does not add any values as a feature
            :param training_file:
            :return:
            '''
            with open(training_file) as file:
                num_points, point_dimensionality = map(int, re.split('\s+', file.readline().strip()))
            file.close()

            # Data is (n_samples, n_features)
            data = np.loadtxt(training_file, skiprows=1, usecols=range(0, point_dimensionality))
            return data, num_points, point_dimensionality

    class BoostIt:
        def __init__(self, context):
            print('initializing')
            self.w⃗ = None
            self.𝛵 = context.ensemble_size
            self.𝓓 = [context.training_files, context.testing_files]
            # self.cardinality_D =
            self.𝒜 = 'triclassify'
            # EnsembleMethod.load_data(self.𝓓.training_file)


        def boosting(self, 𝓓, 𝛵, 𝒜):
            w⃗ = cardinality_D * [float(1)/2]
            # w⃗ = (1 / cardinality_D for x in range(len(eval("{0} * [None]".format(cardinality_D)))))
            print('running boost')
            𝜀 = {}
            𝛼 = {}
            𝛭 = {}
            for t in range(1, 𝛵):
                𝛭[t] = 𝒜(𝓓)  # model M is equal to learning algorithm(data)
                𝜀[t] = 𝛭.weightedError()
                if 𝜀[t] >= float(1)/2:
                   𝛵 = t - 1; break
                𝛼[t] = (float (1)/2) * np.log((1 - 𝜀[t]) / 𝜀[t])
                for x in 𝓓['misclassified']:
                    w⃗[t+1] = w⃗[t]/float(2 * 𝜀[t])
                for x in 𝓓['correctly_classified']:
                    w⃗[t+1] = w⃗[t]/float(2) * (1-𝜀[t])
            return 𝛭




            # class Pseudo:
    # @args: Data type, ensemble size T, learning algorithm A
    # @return: weighted ensemble of models T
    # def __init__(self, data, ensemble_size, learning_algorithm):
        # vec = [float(1) / data.size] * data.size  # Non-lazy eval
        # weight_vec = (1/2 for x in range(data.size))
            # ½ = float(1) / 2
            # ‖


if __name__ == '__main__':
    # weight_vec = [None] * 100
    ctx = EnsembleMethod.Context(sys.argv)
    boostIt = EnsembleMethod.BoostIt(ctx)
    boostIt.boosting()
    # boostIt = BoostIt(sys.argv)

    # cardinality_D = 100
    # ℕ_data_points = 200
    # ⇒, Σ

    # 𝛵: ensemble_size
    # 𝛵 = boostIt.𝛵


    # 𝒜: Learning Algorithm
    # 𝓐 =
    # 𝓓 = data
    # "λ
    # 𝜀
    # 𝛼
    # 𝛵: ensemble_size
    # algFunc = None
    # model = weakModel(algFunc, weight_vec)
    # f = lambda x: sin(x) if 0 <= x <= 2*pi else 0

    # for word in w⃗:
    #     print (word)
    # classifier.runWithContext()


    # ctx = Context(sys.argv)


# /home/angel/Courses/cs165b/machine-learning-cs165b-assignment5/boostit.py