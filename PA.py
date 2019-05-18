import numpy as np
import random
from numpy import linalg as LA

##  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!     THIS IS STEEL SVM, I HAVENT CHANGED YET         !!!!!!!!!!!!!!!!!!!!!!!!!


def PA(x_train,y_train):
    epoches = 10
    class_num = 3 #how to do not hard coded???????
    features_num = len(x_train[0])
    lamda = 0.2
    w = np.zeros((class_num, features_num))
    eta = 0.1
    for e in range(epoches):
        for x, y in zip(x_train, y_train):
            y = int(y)
            # predict
            y_hat = np.argmax(np.dot(w, x))
            #update
            if y != y_hat:
                loss = max(0, 1 - np.dot(w[y:,], np.transpose(x))[0] + np.dot(w[y:, ], np.transpose(x))[0])
                T = loss / (2 * (LA.norm(x)) ** 2)
                w[y, :] = w[y, :] + T * x
                w[y_hat, :] = w[y_hat, :] - T * x
    return w

##  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!     THIS IS STEEL SVM, I HAVENT CHANGED YET         !!!!!!!!!!!!!!!!!!!!!!!!!
