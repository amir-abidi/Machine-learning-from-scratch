from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from knn import KNN
import numpy as np

data = load_iris()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12)
clf: KNN = KNN(k=2)

clf.fit(X_train, y_train)

predictions = clf.predict(X_test)

print(f"accuracy{(np.sum(predictions == y_test )/len(y_test))}")
