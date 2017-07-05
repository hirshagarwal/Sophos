import numpy as np


class Model:
    def __init__(self):
        self.mean = 0
        self.covar = 0

    def predictGaussian(self, mu, covar, x):
        # Returns a prediction for the class of x
        # Take dimension of covariance
        d = covar.shape[0]
        # Put mu and x in column vectors
        mu = mu.T
        x = x.T

        # Evaluate Gaussian
        p = 1 / np.sqrt((2 * np.pi) ** d * np.linalg.det(covar)) * np.exp(
            -0.5 * (x - mu).transpose() * np.linalg.inv(covar) * (x - mu))
        return p

    def mu_class(self, data):
        return np.mean(data, axis=0)

    def mu_data(self, data):
        # TODO - Finish this class
        pass
