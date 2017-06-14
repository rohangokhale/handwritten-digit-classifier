import json
import numpy as np
biases = [["b1"], ["b2", ["b2a"]], ["b3"]]
#jsonBiases = json.dumps([b for b in biases])
jsonBiases = json.dumps([{'biases':biases}])
print jsonBiases