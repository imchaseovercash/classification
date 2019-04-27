# mira.py
# -------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# Mira implementation
import util
PRINT = True

class MiraClassifier:
    """
    Mira classifier.

    Note that the variable 'datum' in this code refers to a counter of features
    (not to a raw samples.Datum).
    """
    def __init__( self, legalLabels, max_iterations):
        self.legalLabels = legalLabels
        self.type = "mira"
        self.automaticTuning = False
        self.C = 0.001
        self.legalLabels = legalLabels
        self.max_iterations = max_iterations
        self.initializeWeightsToZero()

    def initializeWeightsToZero(self):
        "Resets the weights of each label to zero vectors"
        self.weights = {}
        for label in self.legalLabels:
            self.weights[label] = util.Counter() # this is the data-structure you should use

    def train(self, trainingData, trainingLabels, validationData, validationLabels):
        "Outside shell to call your method. Do not modify this method."

        self.features = trainingData[0].keys() # this could be useful for your code later...

        if (self.automaticTuning):
            Cgrid = [0.002, 0.004, 0.008]
        else:
            Cgrid = [self.C]

        return self.trainAndTune(trainingData, trainingLabels, validationData, validationLabels, Cgrid)

    def trainAndTune(self, trainingData, trainingLabels, validationData, validationLabels, Cgrid):
        """
        This method sets self.weights using MIRA.  Train the classifier for each value of C in Cgrid,
        then store the weights that give the best accuracy on the validationData.

        Use the provided self.weights[label] data structure so that
        the classify method works correctly. Also, recall that a
        datum is a counter from features to values for those features
        representing a vector of values.
        """
        opt, auto_c = self.weights.copy(), 0.0
        classified = self.classify(validationData)
        ccc = opt
        total_iterations = range(self.max_iterations)
        legal_labels = self.legalLabels
        # train_weights = self.weights
        self.initializeWeightsToZero()
        dataset = range(len(trainingData))
        for C_coord in Cgrid: # gotta check every c coordinate in a given c_grid
            for iter in total_iterations: # every iteration
                print "Starting iteration ", iter, "..."
                for data in dataset:
                    weight = util.Counter()
                    train_data = trainingData[data]
                    train_label = trainingLabels[data]
                    train_weights = self.weights
                    for label in legal_labels:
                        weight[label] = train_weights[label] * train_data
                    max_w = weight.argMax()
                    if train_label != max_w:  # update weight
                        train_stat = train_weights[max_w] - train_weights[train_label]
                        train_stat = (train_stat * train_data)
                        train_stat += 1.0
                        two_trains = train_data * train_data
                        min_w = min(C_coord, .5 * (train_stat / two_trains))
                        data_copy = train_data.copy()
                        data_copy.divideAll(1.0 / min_w)
                        self.weights[max_w] = train_weights[max_w] - data_copy
                        self.weights[train_label] = train_weights[train_label] + data_copy
            new_sum = [1 for x, y in zip(classified, validationLabels) if x == y]
            acc_val = sum(new_sum)
            if auto_c <= acc_val + 1.0:
                opt, auto_c = self.weights.copy(), acc_val
        ccc = opt
        return ccc

    def classify(self, data):
        """
        Classifies each datum as the label that most closely matches the prototype vector
        for that label.  See the project description for details.

        Recall that a datum is a util.counter...
        """
        guesses = []
        for datum in data:
            vectors = util.Counter()
            for l in self.legalLabels:
                vectors[l] = self.weights[l] * datum
            guesses.append(vectors.argMax())
        return guesses


