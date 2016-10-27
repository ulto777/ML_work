import numpy as np
from sklearn import metrics
from sklearn.cross_validation import train_test_split

data_matrix = np.loadtxt(open('HTRU_2.csv', 'r'), dtype=np.float, delimiter=',', skiprows=0)
data = data_matrix[:, 0:-1]
label = data_matrix[:, 8]


def calculate_w():
    x0 = data[label == 0]
    x1 = data[label == 1]
    mean0 = np.mean(x0, axis=0)
    mean1 = np.mean(x1, axis=0)
    sw = np.zeros((8, 8))
    for i in range(x1.shape[0]):
        xsmean = np.mat(x1[i, :]-mean1)
        sw += xsmean.transpose()*xsmean
    for i in range(x0.shape[0]):
        xsmean = np.mat(x0[i, :]-mean0)
        sw += xsmean.transpose()*xsmean
    w = (mean0-mean1)*(np.mat(sw).I)
    print(w)
    return w

calculate_w()
