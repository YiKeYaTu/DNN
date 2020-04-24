import math
import numpy as np

class Categorical_crossentropy:
    def __init__(self):
        pass
    
    def forward(self, _Y, Y):
        # _Y refers to the prediction, and
        # Y refers to the label
        self._Y = _Y
        self.Y = Y

        loss = 0

        for i in range(len(_Y)):
            loss += - Y[i] * math.log(_Y[i])

        return loss

    def backward(self):
        DZ = np.empty(self._Y.shape, dtype=float)
        for i in range(len(self._Y)):
            DZ[i] = - self.Y[i] / self._Y[i]
        return DZ