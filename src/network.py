"""
network.py
~~~~~~~~~~

A module to implement the stochastic gradient descent learning
algorithm for a feedforward neural network.  Gradients are calculated
using backpropagation.  Note that I have focused on making the code
simple, easily readable, and easily modifiable.  It is not optimized,
and omits many desirable features.
"""

#### Libraries
# Standard library
import random
import time
import json

# Third-party libraries
import numpy as np

from activationfunctions import sigmoid, sigmoidPrime

class NeuralNetwork(object):

    def __init__(self, layerSizes):
        """The list ``layerSizes`` contains the number of neurons in the
        respective layers of the network.  For example, if the list
        was [2, 3, 1] then it would be a three-layer network, with the
        first layer containing 2 neurons, the second layer 3 neurons,
        and the third layer 1 neuron.  The biases and weights for the
        network are initialized randomly, using a Gaussian
        distribution with mean 0, and variance 1.  Note that the first
        layer is assumed to be an input layer, and by convention we
        won't set any biases for those neurons, since biases are only
        ever used in computing the outputs from later layers."""
        self.numLayers = len(layerSizes)
        self.layerSizes = layerSizes
        self.biases = [np.random.randn(y, 1) for y in layerSizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(layerSizes[:-1], layerSizes[1:])]

    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def train(self, trainingData, epochs, miniBatchSize, learnRate, testData=None):
        """Train the neural network using -batch stochastic
        gradient descent.  The ``trainingData`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If ``testData`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially."""
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
        print("Time to train: {0} second.".format(endTime-startTime))
        print("biases:")
        print(self.biases)
        print("weights:")
        print(self.weights)
        
    def updateMiniBatch(self, miniBatch, learnRate):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``miniBatch`` is a list of tuples ``(x, y)``, and ``learnRate``
        is the learning rate."""
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
        """Return a tuple ``(delB, delW)`` representing the
        gradient for the cost function C_x.  ``delB`` and
        ``delW`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        delB = [np.zeros(b.shape) for b in self.biases]
        delW = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.costDerivative(activations[-1], y) * sigmoidPrime(zs[-1])
        delB[-1] = delta
        delW[-1] = np.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in xrange(2, self.numLayers):
            z = zs[-l]
            sPrime = sigmoidPrime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sPrime
            delB[-l] = delta
            delW[-l] = np.dot(delta, activations[-l-1].transpose())
        return (delB, delW)

    def evaluate(self, testData):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        testResults = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in testData]
        return sum(int(x == y) for (x, y) in testResults)

    def costDerivative(self, outputActivations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (outputActivations-y)

    def storeParameters(self, outFilename):
        #print(type(self.biases))
        #print(type(self.weights))
        #pythonData = {'biases': self.biases, 'weights':self.weights}
        #print(type(pythonData))
        #pythonData = dict(pythonData)
        #print(type(pythonData))

        params = {"layerSizes": self.layerSizes,
                    "weights": [w.tolist() for w in self.weights],
                    "biases": [b.tolist() for b in self.biases],
                    }
        with open(outFilename, 'w') as outfile:
            json.dump(params, outfile)
        #with open(outFilename, 'w') as outFile:
        #    json.dump(pythonData, outFile)


    def predict(self, imageData):
        result = np.argmax(self.feedforward(imageData[0]))
        #result = np.argmax(self.feedforward(x)), y for (x, y) in imageData
        print("Prediction: {0}, Actual: {1}".format(result, imageData[1]))
        return result

def loadParameters(paramFilename):
    paramFile = open(paramFilename, 'r')
    params = json.load(paramFile)
    paramFile.close()

    net = NeuralNetwork(params["layerSizes"])
    net.biases = [np.array(bias) for bias in params["biases"]]
    net.weights = [np.array(weight) for weight in params["weights"]]
    return net
