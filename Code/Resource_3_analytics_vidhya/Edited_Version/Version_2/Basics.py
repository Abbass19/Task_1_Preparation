import numpy as np
import math

def formatPrice(n):
    return ("-Rs." if n < 0 else "Rs.") + "{0:.2f}".format(abs(n))

def getStockDataVec(key):
    vec = []
    with open(key + ".csv", "r") as f:
        lines = f.read().splitlines()
    for line in lines[3:]:
        variable = float(line.split(",")[4])  # Closing price column
        vec.append(variable)
    return vec

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def getState(data, t, n):
    d = t - n + 1
    block = data[d : t + 1] if d >= 0 else [data[0]] * (-d) + data[0 : t + 1]
    res = []
    for i in range(n - 1):
        res.append(sigmoid(block[i + 1] - block[i]))
    return np.array([res])
