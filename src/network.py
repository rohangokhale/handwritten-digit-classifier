#network.py
"""
A module to implement the stochastic gradient descent learning
algorithm for a feedforward neural network.  Gradients are calculated
using backpropagation.  Note that I have focused on making the code
simple, easily readable, and easily modifiable.  It is not optimized,
and omits many desirable features.
"""

import random
import time
import json
import numpy as np
from activationfunctions import sigmoid, sigmoidPrime

class NeuralNetwork(object):

    def __init__(self, layerSizes):
        """The constructor takes a list layerSizes and create a network with the
        specified number and size of layers. The first layer is the input layer, 
        and the last is the output layer. All layers in the middle are 'hidden' layers.
        The biases for each layer after the 1st and the weights connecting pairs of
        neurons between layers are initialized with random numbers using a Gaussian
        distributions. The initialized parameters have a mean of zero, and a variance
        of 1."""
        self.numLayers = len(layerSizes)
        self.layerSizes = layerSizes
        self.biases = [np.random.randn(y, 1) for y in layerSizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(layerSizes[:-1], layerSizes[1:])]

    def feedforward(self, a):
        """Takes an input 'a' and feeds it through each layer of the nework, eventually returning
        the output of the output layer."""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def train(self, trainingData, epochs, miniBatchSize, learnRate, testData=None):
        """Trains the neural net with the supplied trainingData and the specfied hyperparameters.
        This is done using a stochastic gradient descent method. testData being supplied will
        evaluate the network after each epoch and print out the accurary of the network at that
        point in time."""
        print("Training network...")
        startTime = int(round(time.time()))
        if testData: nTest = len(testData)
        n = len(trainingData)
        for epochNum in xrange(epochs):
            random.shuffle(trainingData)
            miniBatches = [trainingData[k:k+miniBatchSize] for k in xrange(0, n, miniBatchSize)]
            for miniBatch in miniBatches:
                self.updateMiniBatch(miniBatch, learnRate)
            if testData:
                numCorrect = (self.evaluate(testData))
                print "Epoch {0}: {1} {2} / {3}".format(epochNum+1, float(numCorrect)/nTest*100.0, numCorrect, nTest)
            else:
                print "Epoch {0} complete".format(epochNum)
        fNumCorrect = (self.evaluate(testData))
        print ("Accuracy: {0}: {1} ({2} / {3} correct classifications)".format(epochNum+1, float(numCorrect)/nTest*100.0, numCorrect, nTest))

        endTime = int(round(time.time()))
        print("Time to train: {0} seconds.".format(endTime-startTime))

    def updateMiniBatch(self, miniBatch, learnRate):
        """Takes a mini batch of inputs and the learn rate. Adjusts the weights
        and biases of the network by applying gradient descent, using back propagations
        to the mini batch."""
        delB = [np.zeros(b.shape) for b in self.biases]
        delW = [np.zeros(w.shape) for w in self.weights]
        for x, y in miniBatch:
            delta_delB, delta_delW = self.backprop(x, y)
            delB = [dB+ddB for dB, ddB in zip(delB, delta_delB)]
            delW = [dW+ddW for dW, ddW in zip(delW, delta_delW)]
        self.weights = [w-(learnRate/len(miniBatch))*dW
                        for w, dW in zip(self.weights, delW)]
        self.biases = [b-(learnRate/len(miniBatch))*dB
                       for b, dB in zip(self.biases, delB)]

    def backprop(self, x, y):
        """Returns the graident for the cost function. This is represented
        by the delB and delW values for the biases and weights."""
        delB = [np.zeros(b.shape) for b in self.biases]
        delW = [np.zeros(w.shape) for w in self.weights]
        activation = x
        activations = [x] 
        zs = []
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        delta = self.costDerivative(activations[-1], y) * sigmoidPrime(zs[-1])
        delB[-1] = delta
        delW[-1] = np.dot(delta, activations[-2].transpose())

        for l in xrange(2, self.numLayers):
            z = zs[-l]
            sPrime = sigmoidPrime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sPrime
            delB[-l] = delta
            delW[-l] = np.dot(delta, activations[-l-1].transpose())
        return (delB, delW)

    def evaluate(self, testData):
        """Takes a number of test inputs in testData, evaluates what output
        the neural network gives, and compare the output to the correct
        expected output. The total number of correct predictions is returned."""
        testResults = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in testData]
        return sum(int(x == y) for (x, y) in testResults)

    def costDerivative(self, outputActivations, y):
        """Returns a vector containing the partial derivatives for the
        output activations."""
        return (outputActivations-y)

    def storeParameters(self, outFilename):
        """Writes the parameters (layer sizes, weights, and biases) to file
        with the specified name"""
        params = {"layerSizes": self.layerSizes,
                    "weights": [w.tolist() for w in self.weights],
                    "biases": [b.tolist() for b in self.biases],
                    }
        with open(outFilename, 'w') as outfile:
            json.dump(params, outfile)


    def predict(self, imageData):
        """Takes a single set of image data and returns the predicted digit represented by it."""
        result = np.argmax(self.feedforward(imageData[0]))
        print("Prediction: {0}, Actual: {1}".format(result, imageData[1]))
        return result

def loadNetwork(paramsFilename):
    """Returns a neural network object with the layer details, weights, and biases provided
    in paramsFilename. This file should be json formatted."""
    paramsFile = open(paramsFilename, 'r')
    params = json.load(paramsFile)
    paramsFile.close()
 
    net = NeuralNetwork(params["layerSizes"])
    net.biases = [np.array(bias) for bias in params["biases"]]
    net.weights = [np.array(weight) for weight in params["weights"]]
    return net
