
import mnist_loader
import network


def getLayerCount():
	layerCount = 0
	while layerCount <= 0:
		layerCount = int(input("How many hidden layers would you like in your network? \n \
			(choose a number between 1 and 10): "))
	return layerCount

def getLayerSizes(layerCount):
	layerSizes = []
	for i in range(0, layerCount):
		layerSize = 0
		while layerSize <= 0:
			layerSize = int(input("Enter the number of neurons that hidden layer " + str(i+1) + " should have: "))
		layerSizes.append(layerSize)
	return layerSizes

def getEpochs():
	epochs = 0
	while epochs <= 0:
		epochs = int(input("Enter how many epochs would you like to run during training: "))
	return epochs

def getMiniBatchSize():
	miniBatchSize = 0
	while miniBatchSize <= 0:
		miniBatchSize = int(input("Enter the size of each mini batch: "))
	return miniBatchSize

def getLearnRate():
	learnRate = 0
	while learnRate <= 0:
		learnRate = float(input("Enter the learn rate for the model: "))
	return learnRate

print("Welcome! Let's create and train your neural network to correctly \n \
	classify digits from the MNIST database of images.")

layerCount = getLayerCount()
layerSizes = getLayerSizes(layerCount)
layerSizes.insert(0, 784)
layerSizes.append(10)
net = network.NeuralNetwork(layerSizes)

trainingData, validationData, testData = mnist_loader.load_data_wrapper()
print("Training network...")
net.train(trainingData, getEpochs(), getMiniBatchSize(), getLearnRate(), testData=testData)







#net = network.NeuralNetwork([784, 30, 10])

#net.train(trainingData, 30, 10, 3.0, testData=testData)

