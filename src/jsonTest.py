import json
import numpy as np
biases = [["b1"], ["b2", ["b2a"]], ["b3"]]
weights = [["w1"], ["w2"], ["w3", "w3a"]]
#jsonBiases = json.dumps([b for b in biases])
jsonBiases = json.dumps({'biases':biases}, {'weights':weights})
print jsonBiases