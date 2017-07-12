# Sophos

This is a repository designed to facilitate the understanding of artificial neural networks. The entire framework is written in python so that it is easy to understand how each component works.
With an entire backend written in Python this is far from the most computationally efficient neural network framework available and it is not recommended for use in production models.
For a production ready neural network frameworks consider some of these modern and well maintained frameworks:
* [TensorFlow](https://www.tensorflow.org/)
* [Keras](https://keras.io/)
* [Microsoft Cognitive Toolkit](https://www.microsoft.com/en-us/cognitive-toolkit/)

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

For more information about activation layer please see further documentation: [Understanding Activation Functions](https://github.com/hirshagarwal/Sophos/blob/master/Documentation/Activation.md)

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


### Model Not Training
There are many reasons why a model might not be training.
The most common issue is not running it for long enough or on enough data. Every problem requires a different amount of data depending on its complexity, however usually the network will need to iterate over the entire data set at least a few times. 
Another common issue is oversaturating the network with data that hasn't been normalized. If your data is too large it can oversaturate the sigmoid activation function and stop the model from training.
[Understanding Activation Functions](https://github.com/hirshagarwal/Sophos/blob/master/Documentation/Activation.md)


### Get Prediction
There are multiple methods to get predictions from a model. The two basic implemented functions are predict and predictStep. The only difference is that predictStep run the output through a step function in order to return a binary value.

```python
# Setup data to feed in (should be numpy matrix)
x_data = ...

# Feed in to get an output from the function
model.predict(x_data)

# Feed in to get a binary output
model.predictStep(x_data)
```


### Getting accuracy with epochs
The model object has a built in method to return an error rate, however this error rate corresponds to the error rate of the most recent data point, not the overall accuracy of the model. In order to test the accuracy of the model a test dataset should be isolated and run through the model for training.
It's best to run the testing set through the model regularly in order to observe the accuracy trend. 

The code below is a theoretical example - It is not designed to run in its current form

```python
# Setup some training and test data - All of them should be filled in with a numpy matrix
# The training data should be in a list so each pair can be individually accessed
train_x = ... # Fill in with numpy matrix
train_y = ...

# The test data should be in a list so each pair can be individually accessed
test_x = ...
test_y = ...

# Run for 1000 iterations
for i in range(1000):
	# Train on some data
	model.train(train_x, train_y)

	# Check accuracy if at the end of an epoch
	correct = 0 # Counter for the number of correctly classified points
	if i % 50 == 0:
		for j in range(len(test_data)):
			a = model.predictStep(test_data_x[j]) # Get a prediction for each point of test data
				if a == test_data_y: # Compare the test data to the target
					correct += 1 # If the prediction was correct increment the counter
		accuracy = correct/len(test_data) # This is the accuracy value
```

# Batch Training
** Not yet implemented **