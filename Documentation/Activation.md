# Understanding Activation Functions for Artificial Neural Networks

## What are activation functions
When a neuron in an artificial neural network recieves input it has to make a decision about if it should fire or not. This determination must be made based on a single scalar value - the sum of all the inputs. By applying different activation functions the neuron can decide to output a 0, 1 or any number of continuous values as determined by its activation function.

## Types of activation function
There are a few important activation functions that are commonly used.
* Rectified Linear Unit (ReLU)
* Sigmoid
* Heaviside
* Tanh

### Rectified Linear Unit (ReLU)
The ReLU function is a piecwise defined function:
	f(x) = max(0, x)
One issue with the ReLU function is that it's not differentiable at 0. Also due to the fact that the derivative below 0 is 0 neurons using this activation function can "die" as once the output is 0 no weight changes can be backpropagated and they get stuck in an inactive state.

### Sigmoid
The sigmoid function is one of the most common activation functions. It is defined as:
	1/(1-e^-x)
This function has the advantage of being differentiable at all points. The most common issue with sigmoid functions is oversaturation. When x is large the derivative of the function approaches 0, yielding the same problem as ReLU where at 0 the unit becomes dead and can no longer learn.