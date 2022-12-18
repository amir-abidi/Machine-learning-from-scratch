from typing import List
from collections import Counter
import numpy as np

def euclidean_distance(x1: float, x2: float) -> float:
    """
    This is a method used to calculate the euclidean distances between two points.
    """
    return np.sqrt(np.sum((x2-x1)**2))


class KNN:
    """
    This is a class used to initialize KNN classifier
    from scratch.
    """
    def __init__(self, k: int = 3):
        """
        This is the constructor of the class used to initialize the number of
        neighbors of the
        """
        self.k = k
    
    def fit(self, X, y) -> None:
        """
        This method is used to fit the classifier the given train data.
        """
        self.X_train = X
        self.y_train = y
    
    def predict(self, X:List) :
        """
        This is a method used to predict the class of a given input
        """
        predicted_labels = [self._predict(x) for x in X]
        return np.array(predicted_labels)
        
    def _predict(self, x):
        """
        This is a helper function used to calculate the class of each item."""
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]