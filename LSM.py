import numpy as np
from sklearn import metrics
from sklearn.cross_validation import train_test_split

#read data
data_matrix = np.loadtxt(open('HTRU_2.csv', 'r'), dtype=np.float, delimiter=',', skiprows=0)
data_mat = np.c_[np.ones(data_matrix.shape[0]), data_matrix]
data = data_mat[:, 0:-1]
label = data_matrix[:, 8]

#split data
X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.8, random_state=0)
x_train = np.mat(X_train)
x_test = np.mat(X_test)
w = np.dot(np.linalg.pinv(x_train), y_train)
predicted = np.dot(x_test, w.transpose()) > 0.5
print("Classification report:\n%s\n" % metrics.classification_report(y_test, predicted))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(y_test, predicted))