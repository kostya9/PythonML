import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
# Data 1
# Load data

filename = "ex1data1.txt"
dataset = np.loadtxt(filename, delimiter=",")
X = dataset[:, 0]
Y = dataset[:, 1]

# Do next step because one column = one example

X = np.array([X]).T
Y = np.array(Y)

# Initialise Linear Regression

clf = linear_model.LinearRegression()
clf.fit(X, Y)
linear_model.LinearRegression(fit_intercept=True)

# Check(for 35,000 -> 2798.368764)
# (for 70,000 -> 44554.546310)
# !Note Everything is in 10k
print(clf.predict(3.5), clf.predict(7))
# Plot Data and Predictions

fig, ax = plt.subplots()
predictions = clf.predict(X)
ax.scatter(X, Y)
ax.plot(X, predictions, 'k--')

plt.xlabel('Population of City in 10,000s')
plt.ylabel('Profit in $10,000s')
plt.show()

