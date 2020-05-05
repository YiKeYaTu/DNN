# A simple framework for building artificial neural network

## usage

```python

model = Model()

model.add(Dense(10, input_shape=784))
model.train(x_train, y_train)
model.predict(x_test, y_test)

```