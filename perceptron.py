import numpy as np
from sklearn import metrics
from sklearn.cross_validation import train_test_split

data_matrix = np.loadtxt(open('HTRU_2.csv', 'r'), dtype=np.float, delimiter=',', skiprows=0)
data = data_matrix[:, 0:-1]
label = data_matrix[:, 8]
X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.8, random_state=0)
x_train = np.mat(X_train)
x_test = np.mat(X_test)
# init w b
w = np.zeros((8, 1))
b = 0
# train
step = 0.001
cycle = 500
for j in range(cycle):
    for i in range(x_train.shape[0]):
        yi = int(y_train[i] != 0) * 2 - 1
        result = yi * (np.dot(x_train[i], w) + b)
        if result <= 0:
            feature = np.reshape(x_train[i], (8, 1))
            w += feature * yi * step
            b += yi * step

# test
predict = np.dot(x_test, w) + b > 0

# print result
print("Classification report:\n%s\n" % metrics.classification_report(y_test, predict))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(y_test, predict))