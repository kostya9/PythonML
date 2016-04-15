import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import preprocessing
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

filename = "ex2data2.txt"
dataset = np.loadtxt(filename, delimiter=",")
X = dataset[:, [0,1]]
Y = dataset[:, 2]

plt.plot(X[Y == 1, 0], X[Y == 1, 1], 'ro')
plt.plot(X[Y == 0, 0], X[Y == 0, 1], 'bo')
plt.xlabel('Microchip Test 1')
plt.ylabel('Microchip Test 2')
plt.show()


degree = 6
model = \
    make_pipeline(PolynomialFeatures(degree),
                  linear_model.LogisticRegression(C = 1))
model.fit(X, Y)
h = .02
# Plot the decision boundary. For that, we will assign a color to each
# point in the mesh [x_min, m_max]x[y_min, y_max].
x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure(1, figsize=(4, 3))
plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())

Y_predicted = model.predict(X)
print(np.mean(np.equal(Y_predicted, Y)))
plt.plot(X[Y == 1, 0], X[Y == 1, 1], 'ro')
plt.plot(X[Y == 0, 0], X[Y == 0, 1], 'bo')
plt.plot(X[Y_predicted == 1, 0], X[Y_predicted == 1, 1], 'r+')
plt.plot(X[Y_predicted == 0, 0], X[Y_predicted == 0, 1], 'b+')
plt.xlabel('Microchip Test 1')
plt.ylabel('Microchip Test 2')
plt.show()