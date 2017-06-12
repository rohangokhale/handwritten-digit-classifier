


import numpy as np
import random

from activationfunctions import sigmoid, sigmoidPrime

class NeuralNetwork(object):

	def __init__(self, layerSizes):
		"""note: layer indecies start at 0. So, input layer is layer 0."""
		self.numLayers = len(layerSizes)
		self.layerSizes = layerSizes
		"""create a random bias for each non-input neuron"""
		self.biases = [np.random.randn(y, 1) for y in layerSizes[1:]]
		"""create a random weight for every connection between neurons"""
		self.weights = [np.random.randn(y, x) for x, y in zip(layerSizes[:-1], layerSizes[1:])]

	def feedForward(self, a):
		for bias, weight in zip(self.biases, self.weights):
			a = sigmoid(np.dot(bias, weight) + b)
		return a

	def train(self, trainingData, numEpochs, miniBatchSize, learnRate, testData=None):
		if testData:
			nTest=len(testData)
		n=len(trainingData)
		for epoch in xrange(numEpochs):
			"""create mini batches by shuffling the data and then chopping it up into
			chunks of length miniBatchSize"""
			random.shuffle(trainingData)
			miniBatches = [trainingData[k:(k+miniBatchSize)] for k in range(0, n, miniBatchSize)]
			for batch in miniBatches:
				self.updateMiniBatch(batch, learnRate)
			if testData:
				accuracy = 100.0*self.evaluate(testData)/nTest
				print("Epoch {0}: {1}".format(epoch, accuracy))
			else:
				print("Epoch {0} done".format(epoch))

	"""Apply gradient descent using back propagation"""
	def updateMiniBatch(self, batch, learnRate):
		"""initialize del matrices of proper size to zero"""
		delB = [np.zeros(b.shape) for b in self.biases]
		delW = [np.zeros(w.shape) for w in self.weights]
		for x, y in batch:
			del2B, del2W = self.backprop(x, y)
			delB = [dB+d2B for dB, d2B in zip(delB, del2B)]
			delW = [dW+d2W for d2, d2W in zip(delW, del2B)]
		self.biases = [b-(learnRate/len(batch))*dB for b, dB in zip(self.biases, delB)]
		self.weights = [w-(learnRate/len(batch))*dW for w, dW in zip(self.weights, delW)]
		
	def backprop(self, x, y):
		delB = [np.zeros(b.shape) for b in self.biases]
		delW = [np.zeros(w.shape) for w in self.weights]

		activation = x
		activations = [x]
		zs = []
		"""Calculate and store activations"""
		for b, w in zip(self.biases, self.weights):
			z = np.dot(w, activation)+b
			zs.append(z)
			activation = sigmoid(z)
			activations.append(activation)

		"""activations[-1] = the output of the last layer
		"""
		delta = self.costDerivative(activations[-1], y) * sigmoidPrime(zs[-1])
		delB[-1] = delta
		delW[-1] = np.dot(delta, activations[-2].transpose())

		"""Work backwards from second to last layer to the first non-input layer"""
		for l in xrange(self.numLayers-2, 0, -1):
			z = zs[l]
			sPrime = sigmoidPrime(z)
			print("Weights length = {0}".format(len(self.weights)))
			print(l+1)
			delta = np.dot(self.weights[l+1].transpose(), delta) * sPrime
			delB[l] = delta
			delW[l] = np.dot(delta, activations[l-1].transpose())

		return(delB, delW)

	"""Note: the final output of the neural network is the number associated with the neuron
	that has the highest activation."""
	def evaluate(self, testData):
		testResults = [(np.argmax(self.feedForward(x)), y) for (x, y) in testData]
		return sum(int(x==y) for (x,y) in testResults)

	def costDerivative(self, outputActivations, y):
		return(outputActivations-y)

