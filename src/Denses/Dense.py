import numpy as np
from Denses.Neuron import Neuron

class Dense:
    def __init__(self, neurons_count, input_shape = None):
        self.input_shape = input_shape
        self.output_shape = neurons_count
        self.neurons_count = neurons_count
        self.neurons = []
        self.layer = 1

    def __init_neurons__(self):
        for i in range(self.neurons_count):
            self.neurons.append(Neuron(input_shape=self.input_shape))

    def init(self, prev_dense=None):
        if prev_dense != None:
            self.input_shape = prev_dense.output_shape
            self.layer = prev_dense.layer + 1
        self.__init_neurons__()

    def forward(self, x):
        Y = np.empty((self.neurons_count, ), dtype=float)
        for i in range(len(self.neurons)):
            Y[i] = self.neurons[i].forward(x)

        return Y

    def backward(self, DZ):
        NDZ = np.empty((self.neurons_count, self.input_shape), dtype=float)
        
        for i in range(len(self.neurons)):
            NDZ[i] = self.neurons[i].backward(DZ[i])

        NDZ = NDZ.sum(axis=0)

        return NDZ
