"""Definitions for activation functions and their derivatives"""

import numpy as np

def sigmoid(z):
	return 1.0/(1.0+np.exp(-z))

def sigmoidPrime(z):
	return sigmoid(z)*(1-sigmoid(z))


