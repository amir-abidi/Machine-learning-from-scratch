import numpy as np
from typing import List

class LinearRegression:
    """
    This class is used to implement the Linear Regression algorithm
    """
    def __init__(self, lr: float = 0.0001, n_iters: int = 1000):
        """
        This is the constructor of the class used to initialize the class variables.
        """
        self.lr: float = lr
        self.weights = None
        self.bias: float = None
        self.n_iters: int = n_iters
    
    def fit(self, X, y) -> None:
        """
        This method is used to train the model
        """
        n_samples, n_features = X.shape
        self.bias: float = 0
        self.weights = np.zeros(n_features)
        for _ in range(self.n_iters):
            y_predicted = np.dot(X, self.weights) + self.bias
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
    
    def predict(self, x: int | float) -> float:
        """
        This method is used to predict the output for the given input
        """
        return np.dot(x, self.weights) + self.bias
