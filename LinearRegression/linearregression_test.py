from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression
import numpy as np
from linearregression import LinearRegression

def mse(y_true, y_pred) -> float:
    """
    This function is used to calculate the mean squared error.
    """
    return np.mean((y_true - y_pred)**2)

X, y = make_regression(n_samples= 1000, n_features=1, noise = 60, random_state=4)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 12)
regressor = LinearRegression(lr = 0.001, n_iters = 1000)
regressor.fit(X_train, y_train)
predictions = regressor.predict(X_test)

print(mse(y_test, predictions))