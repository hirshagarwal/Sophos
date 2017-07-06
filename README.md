# Sophos

This is a repository designed to facilitate the understanding of artificial neural networks. The entire framework is written in python so that it is easy to understand how each component works.

## General Structure
The general use of Sophos is based on the Keras framework. Different objects can be created and added to the model to form a network for data to flow through

## Layers
There are two types of objects currently available - Fully Connected Layers and Activation Layers
Fully connected layers are feed forward layers in which the input size and number of neurons are specified.
Activation Layers specify an activation function. Currently the implemented functions are Heaviside, ReLU and Sigmoid.

Example:
```python
l1 = Layer()
```

## Models
