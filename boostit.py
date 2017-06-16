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

def load_data(training_file, is_supervised_data=False):
    '''
    Dynamically transform a data file into a numpy ndarray with any N features.
    The first column is ignored because the record number does not add any values as a feature
    :param training_file:
    :return:
    '''
    with open(training_file) as file:
        num_points, point_dimensionality = map(int, re.split('\s+', file.readline().strip()))
    file.close()

    # Data is (n_samples, n_features)

    if is_supervised_data:
        data = np.loadtxt(training_file, skiprows=1, usecols=range(0, point_dimensionality + 1))
        predictors = data[:,0:-1]
        labels = data[:,-1]
        return predictors, labels, num_points, point_dimensionality

    else:
        data = np.loadtxt(training_file, skiprows=1, usecols=range(0, point_dimensionality))
        return data, num_points, point_dimensionality


def compute_centroid(data):
    '''
    :param data:
    :return: mean numpy array of column means
    '''
    # mean = data.sum(axis=0) / float(len(data))  # element wise divide
    mean = np.mean(data, axis=0)
    return mean

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
            self.training_predictors = np.concatenate((self.pos_training_predictors, self.neg_training_predictors), axis=0)

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

        def _load_data(training_file, is_supervised_data=False):
            '''
            Dynamically transform a data file into a numpy ndarray with any N features.
            The first column is ignored because the record number does not add any values as a feature
            :param training_file:
            :return:
            '''
            with open(training_file) as file:
                num_points, point_dimensionality = map(int, re.split('\s+', file.readline().strip()))
            file.close()

            # Data is (n_samples, n_features)

            if is_supervised_data:
                data = np.loadtxt(training_file, skiprows=1, usecols=range(0, point_dimensionality + 1))
                predictors = data[:, 0:-1]
                labels = data[:, -1]
                return predictors, labels, num_points, point_dimensionality

            else:
                data = np.loadtxt(training_file, skiprows=1, usecols=range(0, point_dimensionality))
                return data, num_points, point_dimensionality

        def loadSimpData(self):
            self.datMat = np.matrix([[1., 2.1],
                             [2., 1.1],
                             [1.3, 1.],
                             [1., 1.],
                             [2., 1.]])
            self.classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
            return self.datMat, self.classLabels

    class BoostIt:
        def __init__(self, context):
            self.w⃗ = None
            self.𝛵 = context.ensemble_size
            self.𝓓 = [context.training_files, context.testing_files]
            # self.cardinality_D =
            self.𝒜 = 'triclassify'
            # EnsembleMethod.load_data(self.𝓓.training_file)


        def boosting(self, 𝓓, 𝛵, 𝒜):
            w⃗ = len(𝓓) * [float(1)/2]
            # w⃗ = (1 / cardinality_D for x in range(len(eval("{0} * [None]".format(cardinality_D)))))
            𝜀 = {}
            𝛼 = {}
            𝛭 = {}
            for t in range(1, 𝛵):
                𝛭[t] = 𝒜(𝓓)  # model M is equal to learning algorithm(data)
                𝜀[t] = 𝛭.weightedError()
                if 𝜀[t] >= float(1)/2:
                   𝛵 = t - 1; break
                𝛼[t] = (float (1)/2) * np.log((1 - 𝜀[t]) / max(𝜀[t],1e-16))
                for x in 𝓓['misclassified']:
                    w⃗[t+1] = w⃗[t]/float(2 * 𝜀[t])
                for x in 𝓓['correctly_classified']:
                    w⃗[t+1] = w⃗[t]/float(2) * (1-𝜀[t])
            return 𝛭

def trainEnsemble(weights):
    global clusterPosTrain, clusterNegTrain
    top = np.array(np.array(weights[0:int(len(weights)/2)]))
    bottom = np.array(weights[int(len(weights)/2):])
    clusterPosTrain = []
    clusterNegTrain = []
    for i in range(ctx.pos_training_dimensionality):
        clusterPosTrain.append(compute_centroid(ctx.pos_training_predictors[:,i]*top))
        clusterNegTrain.append(compute_centroid(np.multiply(ctx.neg_training_predictors[:,i],bottom)))
    clusterPosTrain = np.array(clusterPosTrain)
    clusterNegTrain = np.array(clusterNegTrain)

    varianceTrain = np.subtract(clusterPosTrain, clusterNegTrain)
    W_train = (clusterPosTrain - clusterNegTrain)
    T_train = 0.5 * np.inner(clusterPosTrain + clusterNegTrain, (clusterPosTrain - clusterNegTrain))
    T_train_array = np.full(ctx.num_training, fill_value=T_train, dtype=float)

    clusterPosTest = []
    clusterNegTest = []
    top = np.array(np.array(weights[0:int(len(weights) / 2)]))
    bottom = np.array(weights[int(len(weights) / 2):])
    for i in range(ctx.pos_training_dimensionality):
        clusterPosTest.append(compute_centroid(ctx.pos_testing_predictors[:,i]*top))
        clusterNegTest.append(compute_centroid(np.multiply(ctx.neg_testing_predictors[:,i],bottom)))
    clusterPosTest = np.array(clusterPosTest)
    clusterNegTest = np.array(clusterNegTest)

    W_test = (clusterPosTest - clusterNegTest)

    testSample = np.inner(ctx.testing_predictors, W_test) - T_train
    X_test_diff = testSample - T_train_array
    errors_vector = X_test_diff < 0
    errors = errors_vector.astype(int)
    error = float(sum(errors)) / ctx.num_training
    alpha = 0.5 * np.log((1.0 - error) / max(error,1e-16))

    return weights, error, errors_vector, alpha

def boost(margin_error, errors_vector, weights, alpha):
    for i, error in enumerate(errors_vector):
        if error:
            alpha = 0.5 * np.log((1.0 - margin_error) / max(margin_error, 1e-16))
            weights[i] += alpha
        else:
            alpha = 0.5 * np.log((float(margin_error)) / max(margin_error, 1e-16))
            weights[i] -= alpha
    return weights

if __name__ == '__main__':
    # weight_vec = [None] * 100
    ctx = EnsembleMethod.Context(sys.argv)
    weights = np.full(ctx.num_training, fill_value=1, dtype=int)

    for i in range(int(ctx.ensemble_size)):
        weights, error, errors_vector, alpha = trainEnsemble(weights)
        print("Iteration {0}".format(i))
        print("Error = {0:.2f}".format(error))
        print("Alpha = {0:.2f}".format(alpha))
        print("Factor to increase weights = {0:.2f}".format(alpha))
        print("Factor to decrease weights = {0:.2f}".format(alpha))
        boost(error, errors_vector, weights, alpha)

    clusterPosTest = []
    clusterNegTest = []
    top = np.array(np.array(weights[0:int(len(weights) / 2)]))
    bottom = np.array(weights[int(len(weights) / 2):])
    for i in range(ctx.pos_training_dimensionality):
        clusterPosTest.append(compute_centroid(ctx.pos_testing_predictors[:, i] * top))
        clusterNegTest.append(compute_centroid(np.multiply(ctx.neg_testing_predictors[:, i], bottom)))
    clusterPosTest = np.array(clusterPosTest)
    clusterNegTest = np.array(clusterNegTest)

    W_test = (clusterPosTest - clusterNegTest)
    T_test = 0.5 * np.inner(clusterPosTest + clusterNegTest, (clusterPosTest - clusterNegTest))
    T_test_array = np.full(ctx.num_testing, fill_value=T_test, dtype=float)

    testSample = np.inner(ctx.testing_predictors, W_test) - T_test
    X_test_diff = testSample - T_test_array
    errors_test_vector = X_test_diff < 0
    correct_test_vector = np.array(X_test_diff > 0)
    errors_test = errors_test_vector.astype(int)
    error = float(sum(errors_test)) / ctx.num_training

    ground_truth_targets = np.array(ctx.y_testing_targets)

    # pos_hits = np.logical_not(np.logical_xor(predicted, ground_truth_targets))
    false_positives = 0
    false_negatives = 0
    for i in range(len(errors_test_vector)):
        if errors_test[i] and ground_truth_targets[i] == 1:
            false_positives += 1
        if errors_test[i] and ground_truth_targets[i] == -1:
            false_negatives += 1
    print('False positives:\t{0}'.format(false_positives))
    print('False negatives:\t{0}'.format(false_negatives))
    error_rate = sum(errors_test) / float(ctx.num_testing)
    print('Error rate:\t{0}%'.format(int(error_rate * 100)))
