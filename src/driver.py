
import mnist_loader
import network


print("Welcome! Let's create and train your neural network to correctly \n \
	classify digits from the MNIST database of images.")

layerCount = getLayerCount()
layerSizes = getLayerSizes()
print layerSizes


"""
trainingData, validationData, testData = mnist_loader.load_data_wrapper()


net = network.NeuralNetwork([784, 30, 10])
net.train(trainingData, 30, 10, 3.0, testData=testData)
"""

def getLayerCount():
	layerCount = 0
	while layerCount <= 0:
		layerCount = int(input("How many hidden layers would you like in your network? \n \
			(choose a number between 1 and 10"))
	return layerCount

def getLayerSizes(layerCount):
	layerSize = 0
	layerSizes = []
	for i in range(1:layerCount):
		while layerSize <= 0:
			layerSize = int(input("Enter the number of neurons that layer " + (i+1) + "should have)"))
		layerSizes.append(layerSize)
	return layerSizes