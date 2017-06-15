import json
import numpy as np
biases = [["b1"], ["b2", ["b2a"]], ["b3"]]
weights = [[5], ["w2"], ["w3", 3]]
#jsonBiases = json.dumps([b for b in biases])
pythonData = {'biases': biases, 'weights':weights}
jsonData = json.dumps([{'biases':biases}, {'weights':weights}])

print(pythonData['biases'])
print(pythonData['weights'])

with open('params.json', 'w') as outFile:
	json.dump(pythonData, outFile)

with open('params.json', 'r') as inFile:
	loadedParams = json.load(inFile)

print loadedParams['biases']
print loadedParams['weights']
#print jsonData