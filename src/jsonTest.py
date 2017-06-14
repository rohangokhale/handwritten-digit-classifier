import json
biases = [["b1"], ["b2"], ["b3"]]
jsonBiases = json.dumps([b for b in biases])
print jsonBiases