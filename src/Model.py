import numpy as np
from Denses.Dense import Dense
from Activations.Softmax import Softmax
from Losses.Categorical_crossentropy import Categorical_crossentropy
from Validations.classify import classify

class Model:
    def __init__(self):
        self.denses = []

    def add(self, dense):
        dense_count = len(self.denses)
        if dense_count > 0:
            dense.init(self.denses[dense_count - 1])
        else:
            dense.init()
        
        self.denses.append(dense)
    
    def train(self, X, Y, batch_size = 1, epochs = 1, loss = "categorical_crossentropy"):
        loss_fn = Categorical_crossentropy()
        softmax_fn = Softmax()

        total = 0
        correct_count = 0

        for epoch in range(epochs):
            for i in range(len(X)):
                # Start forward computing
                x = X[i]

                nx = x
                for dense in self.denses:
                    nx = dense.forward(nx)
                    
                y = softmax_fn.forward(nx)
                loss_val = loss_fn.forward(y, Y[i])

                total += 1
                correct_count += classify(y, Y[i])

                print("epoch: ", epoch, "======== Loss: ", loss_val, ", Accuracy: ", correct_count / total * 100, "%")

                # Start backward computing
                dz = loss_fn.backward()
                dz = softmax_fn.backward(dz)

                for dense in reversed(self.denses.copy()):
                    dz = dense.backward(dz)

    def predict(self, X, Y):
        loss_fn = Categorical_crossentropy()
        softmax_fn = Softmax()

        total = 0
        correct_count = 0

        for i in range(len(X)):
            # Start forward computing
            x = X[i]

            nx = x
            for dense in self.denses:
                nx = dense.forward(nx)
                
            y = softmax_fn.forward(nx)
            loss_val = loss_fn.forward(y, Y[i])

            total += 1
            correct_count += classify(y, Y[i])

            print("Prediction: ======== Loss: ", loss_val, ", Accuracy: ", correct_count / total * 100, "%")
