import numpy as np
from sklearn import svm
from sklearn import metrics
from sklearn.cross_validation import train_test_split

#read data
data_matrix = np.loadtxt(open('HTRU_2.csv', 'r'), dtype=np.float, delimiter=',', skiprows=0)
data = data_matrix[:, 0:8]
label = data_matrix[:, 8]

#split data
X_train, X_test, Y_train, Y_test = train_test_split(data, label, test_size=0.8, random_state=0)

#train and predict with svm
clf = svm.SVC(gamma=0.001)
clf.fit(X_train, Y_train)
predicted = clf.predict(X_test)

#print results
print("Classification report for classifier :\n%s\n"
              % metrics.classification_report(Y_test, predicted))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(Y_test, predicted))