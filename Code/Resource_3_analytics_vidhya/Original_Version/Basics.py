import numpy as np
import math



def formatPrice(n):
    return("-Rs." if n<0 else "Rs.")+"{0:.2f}".format(abs(n))
def getStockDataVec(key):

    #This work here is simplification Work taking into consideration only one parameter. Open
    #In my model we should take all the features into consideration.
    vec = []
    lines = open(key+".csv","r").read().splitlines()
    for line in lines[3:]:
        #print(line)
        #print(float(line.split(",")[4]))
        variable = (float(line.split(",")[4]))
        print(variable)
        vec.append(float(line.split(",")[4]))
        #print(vec)
    return vec
def sigmoid(x):
    return 1/(1+math.exp(-x))



def getState(data, t, n):
    #This function returns a vector. That gives a reading for the relative change of stock mapped through sigmoid [0,1].
    #   if the value is less than 0.5 then the stock has decreased
    #   if the value is more than 0.5 then the stock has increased
    d = t - n + 1
    block = data[d:t + 1] if d >= 0 else -d * [data[0]] + data[0:t + 1] # pad with t0
    res = []
    for i in range(n - 1):
        res.append(sigmoid(block[i + 1] - block[i]))
    return np.array([res])
