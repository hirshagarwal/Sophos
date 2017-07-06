'''
	This program will train itself to learn a linear classification based on the line y=x
	It is trained using only two data points, so the exact line that the model learns is not necessarily going to fit the data particularly well
	The model will however be sufficient to learn and classify our two data points

	In order to make the model trainable the activation layer used has a sigmoid function - Although it's not the best for approximating a known straight line it will allow
	the model to be trained automatically using gradient descent
'''

# Import requirements
import numpy as np
import sys, os
# Move the working director up a folder to import Sophos library
sys.path.insert(0, os.path.abspath('..'))
from Sophos import SophosNet as sn

# Create empty model
model = sn.Model()

# Make a layer with input size of two and one neuron
l1 = sn.Layer(2, 1)

# Make an activation layer with a sigmoid activation function
a1 = sn.Layer('sigmoid')

# Add the two layers to the model
model.add(l1)
model.add(a1)

# Set the model learning rate - By default it's 0.1
model.setLearningRate(0.1)