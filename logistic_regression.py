import numpy as np
from sklearn import metrics

class LogisticRegression:
    def __init__(self):
        self.features = np.matrix('0')
        self.labels = np.matrix('0')
        self.weights = np.matrix('0')
        self.test_set = np.matrix('0')
        data_matrix = np.loadtxt(open('HTRU_2.csv', 'r'), dtype=np.float, delimiter=',')
        w0 = np.ones(data_matrix.shape[0])
        self.data_mat = np.c_[w0, data_matrix]

    #1.split data with defined radio
    def split_data(self, train_radio):
        m, n = self.data_mat.shape
        train_len = int(m * train_radio)
        self.test_set = self.data_mat[train_len:]
        train_set = self.data_mat[0:train_len]
        self.features = train_set[:, 0:-1]
        self.labels = train_set[:, -1:]

    #2.split data with pos/neg radio
    def split_train_data(self, pos_radio, train_radio):
        m, n = self.data_mat.shape
        positive_mat = np.mat([row for row in self.data_mat if row[n - 1] > 0.5])
        negative_mat = np.mat([row for row in self.data_mat if row[n - 1] < 0.5])
        print(positive_mat.shape)
        print(negative_mat.shape)
        train_len = int(m * train_radio)
        pos_num = int(train_len * pos_radio)
        pos_all = positive_mat.shape[0]
        if pos_num > pos_all:
            print("positive data are not enough")
        else:
            train_set = np.r_[positive_mat[0:pos_num, :], negative_mat[0:(train_len-pos_num), :]]
            self.features = train_set[:, 0:-1]
            self.labels = train_set[:, -1:]
            self.test_set = np.r_[positive_mat[pos_num:, :], negative_mat[(train_len-pos_num):, :]]

    @staticmethod
    def sigmoid(x):
        return 1.0/(1.0 + np.exp(-x))

    def grad_ascent(self, alpha=0.001):
        m, n = self.features.shape
        max_cycle = 600
        self.weights = np.mat(np.ones((n, 1)))  # Initialize the weight vector
        sig = np.vectorize(self.sigmoid)
        for k in range(max_cycle):
            h = sig(self.features * self.weights)
            error = h - self.labels
            self.weights -= (alpha / m) * self.features.transpose() * error

    def predict(self):
        features = self.test_set[:, 0:-1]
        labels = self.test_set[:, -1:]
        predicted = np.dot(features, self.weights) > 0.5
        print("Classification report for logistic regression :\n%s\n"
              % metrics.classification_report(labels, predicted))
        print("Confusion matrix:\n%s" % metrics.confusion_matrix(labels, predicted))

cls = LogisticRegression()
#cls.split_data(train_radio=0.8)
cls.split_train_data(pos_radio=0.25, train_radio=0.15)
cls.grad_ascent()
cls.predict()