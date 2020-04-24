import numpy as np
import math

class Softmax:
    def __init__(self):
        pass
    
    def forward(self, X):
        self.base = 0

        for x in X:
            self.base += np.exp(x)

        self.Y = np.empty(X.shape)

        for i in range(len(self.Y)):
            self.Y[i] = np.exp(X[i]) / self.base

        return self.Y

    def backward(self, DZ):
        NDZ = np.zeros(DZ.shape, dtype=float)

        for i in range(len(NDZ)):
            for j in range(len(NDZ)):
                if i == j:
                    NDZ[i] += DZ[i] * self.Y[i] * (1 - self.Y[i])
                else:
                    NDZ[i] += - DZ[j] * self.Y[i] * self.Y[j]
        
        return NDZ
