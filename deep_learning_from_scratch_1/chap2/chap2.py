import numpy as np

def sigmoid(x) :
    return 1.0 / (1.0 + np.exp(-x))


print(sigmoid(1.0))
print(sigmoid(-1000.0))