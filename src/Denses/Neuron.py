import numpy as np

class Neuron:
    def __init__(self, input_shape, learning_rate = 0.01):
        self.input_shape = input_shape
        self.weights = (-1 + 2 * np.random.rand(input_shape)) / 100
        self.bias = 0
        self.learning_rate = learning_rate

    def forward(self, x):
        self.x = x
        self.y = np.dot(x, self.weights) + self.bias

        return self.y

    def backward(self, dz):
        dw = dz * self.x
        db = dz * 1

        ndz = dz * self.weights

        self.weights -= self.learning_rate * dw
        self.bias -= self.learning_rate * db
        # For previous neurons to operate derviate
        return ndz
