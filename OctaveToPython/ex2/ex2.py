import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

# Data 1
# Load data

filename = "ex2data1.txt"
dataset = np.loadtxt(filename, delimiter=",")
X = dataset[:, [0,1]]
Y = dataset[:, 2]
"""
# Plotting

plt.plot(X[Y == 1, 0], X[Y == 1, 1], 'ro')
plt.plot(X[Y == 0, 0], X[Y == 0, 1], 'bo')
plt.xlabel('Exam 1 score')
plt.ylabel('Exam 2 score')
plt.show()
"""
# Logistic regression
# REMEMBER that C = 1/lambda!!!!
clf = linear_model.LogisticRegression(C=1e5, solver = "liblinear")
clf.fit(X,Y)

# Plotting

# TODO Have noa little idea how does this work
#h - step
h = .02
# Plot the decision boundary. For that, we will assign a color to each
# point in the mesh [x_min, m_max]x[y_min, y_max].
x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure(1, figsize=(4, 3))
plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
"""
This is removing axis nmbers
plt.xticks(())
plt.yticks(())
"""
Y_predicted = clf.predict(X)
print(np.mean(np.equal(Y_predicted, Y)))
plt.plot(X[Y == 1, 0], X[Y == 1, 1], 'ro')
plt.plot(X[Y == 0, 0], X[Y == 0, 1], 'bo')
plt.plot(X[Y_predicted == 1, 0], X[Y_predicted == 1, 1], 'r+')
plt.plot(X[Y_predicted == 0, 0], X[Y_predicted == 0, 1], 'b+')
plt.xlabel('Exam 1 score')
plt.ylabel('Exam 2 score')
plt.show()
#Should be 0.776289
print(clf.predict_proba([45, 85]))