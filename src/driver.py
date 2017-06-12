
import mnist_loader
trainingData, validationData, testData = mnist_loader.load_data_wrapper()

import network
net = network.NeuralNetwork([784, 30, 10])
net.train(trainingData, 30, 10, 3.0, testData=testData)