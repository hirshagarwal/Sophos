
# coding: utf-8

# In[399]:


# Setup import
import numpy as np


# In[524]:


""" Model Class
    Contains a linear sequence of all the layers in the model
"""
class Model():
    
    # Model will contain sequence of layers to feed data 
    def __init__(self):
        self.components = list()
    
    # Add a layer to the model
    def add(self, component):
        self.components.append(component)
        
    # Feed Data and Learn
    def train(self, X, Y):
        lastOut = X
        # Feed Data
        for layer in self.components:
            lastOut = layer.feed(lastOut)
        # Set the prediction
        pred = lastOut
        
        # Verify that output layer is an activation function
        if type(self.components[len(self.components)-1]) != Activation:
            raise ValueError("Output layer must have activation function")
        
        # Optimize
        components = self.components
        # Get error for output layer
        error = list()
        error.append(np.sum((Y - pred) * components[len(components)-1].d_feed(pred)))
        for i in range(len(components)-2, 0, -1):
            component = components[i]
            if type(component) == Layer:
                # Error for hidden layers
                error.append(np.sum((component.getWeights() * error[0]) * components[i+1].d_feed(component.getOutput()))) 
        print("Error:", error)
        
        return lastOut
    
    def show(self):
        model_display = ""
        for component in self.components:
            model_display += 'Type: {}\n'.format(component.getType())
        model_display += "---------------"
        return model_display

    def predict(self, X):
        pass


# In[525]:


class Layer():
    
    def __init__(self, num_neurons, num_inputs):
        self.num_neurons = num_neurons
        self.num_inputs = num_inputs
        # Build Psi - Random Weights
        self.W = np.random.random_sample((num_inputs + 1, num_neurons))
        self.last_output = 0
        
    def feed(self, X):
        biases = np.ones(X.shape[0])
        # Add bias
        X = np.insert(X, 0, biases, axis=1)
        # Error checking
        if X.shape[1] != self.W.shape[0]:
            raise ValueError("Wrong input shape")
            
        # Remember Last Output
        self.last_output = X * self.W
        
        # Multiply
        return X * self.W
        
    def getWeights(self):
        return self.W
    
    def setWeights(self, X):
        # Set all of the weights to a new value
        self.W = X
        
    def updateWeights(self, X):
        # Adjust Weights by a value
        pass
    
    def getOutput(self):
        return self.last_output
    
    def getType(self):
        return "Dense Layer"


# In[526]:


class Activation():
    
    # Activation Functions
    def __init__(self, activation):
        self.activation_function = activation
    
    def feed(self, X):
        if self.activation_function == 'step':
            return np.piecewise(X, [X < 0, X >= 0], [0, 1])
        if self.activation_function == 'sigmoid':
            return 1/(1 + np.exp(-X))
        if self.activation_function == 'relu':
            np.maximum(X, 0, X)
            return X
    def d_feed(self, X):
        if self.activation_function == 'sigmoid':
            return X * (1-X)
        if self.activation_function == 'relu':
            return np.piecewise(X, [X < 0, X >= 0], [0, 1])
        
    def getType(self):
        return "Activation Function {fxn}".format(fxn=self.activation_function)


# In[529]:


class Optimizer():
    
    def __init__(self, lr):
        self.lr = lr
    
    # Define Optimizers
    def grad(self, X, Y):
        # Define error and loss
        error = pred-Y
        loss = np.sum(error ** 2)
        
        # Calculate Gradient
        gradient = X.T.dot(error) / X.shape[0]
        
        return gradient