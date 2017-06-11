


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

	def train(self, trainingData, numEpochs, miniBatchSize, learnRate, validateData=None):
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
			if validateData:
				accuracy = self.evaluate(testData)/nTest
				print("Epoch {0}: {1}".format(epoch, accuracy)
			else:
				print("Epoch {0} done".format(epoch)

	"""Apply gradient descent using back propagation"""
	def updateMiniBatch(self, batch, learnRate):
		



