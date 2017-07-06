# Sophos

This is a repository designed to facilitate the understanding of artificial neural networks. The entire framework is written in python so that it is easy to understand how each component works.

## General Structure
The general use of Sophos is based on the Keras framework. Different objects can be created and added to the model to form a network for data to flow through
Numpy and the Sophos Net library are both important things to import.

Import the library
```python
import SophosNet as sn
import numpy as np
```

## Layers
There are two types of objects currently available - Fully Connected Layers and Activation Layers
Fully connected layers are feed forward layers in which the input size and number of neurons are specified. On creation the weights of a fully connected layer are randomly initialized.
Activation Layers specify an activation function. Currently the implemented functions are Heaviside, ReLU and Sigmoid.

Example:
```python
# Create a fully connected layer
l1 = sn.Layer()
# Create an activation layer with a Sigmoid activation
a1 = sn.Activation('sigmoid')
# Create an activation layer with a ReLU activation
a2 = sn.Activation('relu')
# Create an activation layer with a Heaviside activation
a3 = sn.Activation('step')
```

## Models
A model contains a sequence of layers. When you train or run the model it feeds data from layer to layer in order to generate an output.

```python
# Create a new model
model = sn.Model()
# Add layers created before to the model
model.add(l1)
model.add(a1)
```

### Training a model
Models train almost entirely by themselves. Each time the train method runs it updates the weights. Putting it in a for loop is usually the best way to run it a number of times. The data input should be a numpy matrix.
In order to train a model each fully connected layer must be follwed by a sigmoid or relu activation layer - Because the step function is not differentiable it's not compatible with gradient descent.

```python
# Create example data to feed
x_input = np.matrix('1 1')
y_input = np.matrix('0')

# Train the model 1000 times
for i in range(1000):
	model.train(x_input, y_input)
```