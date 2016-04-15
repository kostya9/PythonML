from sklearn import linear_model
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import display_data_func as d
import get_image
from sklearn.externals import joblib
from sklearn.cross_validation import train_test_split


# Display And Load Data

mat_contents = sio.loadmat("ex3data1.mat")

X = mat_contents['X']
Y = mat_contents['y']
Y = np.array(Y)
X = np.array(X)


X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
                                                    test_size=0.2,
                                                    random_state=0)
display_example_count = 12
ixs = np.random.randint(0, 4999, display_example_count)
d.display_data(X[ixs, :], display_example_count)

# Train
logistic = linear_model.LogisticRegression(C = 1e05, multi_class="ovr")
Y_train_r = np.array(Y_train).ravel()
logistic.fit(X_train, Y_train_r)

# Test

Y_predicted = logistic.predict(X_train)
Y_predicted = np.array(Y_predicted)
Y_train = np.array(Y_train)


success_train = np.mean(np.equal(Y_predicted.ravel(), Y_train.ravel()))
success_test = np.mean(np.equal(Y_test.ravel(), logistic.predict(X_test)))

print("The prediction rate for test examples is %i %%" % (float(success_test) * 100))

print("The prediction rate for train examples is %i %%" % (float(success_train) * 100))
joblib.dump(logistic, 'logistic_trained_digits.pkl')
print("Trained logistic regression successfully saved in logistic_trained_digits.pkl")
"""
# Interactive part
display_example_count = 10
ixs = np.random.randint(0, 4999 * 0.2, display_example_count)
"""
"""
dataset_X = X_test[ixs, :]
dataset_Y = logistic.predict(dataset_X)
d.display_data(dataset_X, display_example_count)
"""
