
import mnist_loader
import network

"""
print("Welcome.")
print("How many hidden layers would you like ")
"""


trainingData, validationData, testData = mnist_loader.load_data_wrapper()


net = network.NeuralNetwork([784, 30, 10])
net.train(trainingData, 30, 10, 3.0, testData=testData)