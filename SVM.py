import numpy as np
import random


def SVM(x_train,y_train):
    epoches = 400
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
                w[y, :] = w[y, :] * (1 - (eta*lamda)) + eta * x
                w[y_hat, :] = w[y_hat, :] * (1 - (eta*lamda)) - eta * x
            else:
                w[y, :] = w[y, :] * (1 - (eta * lamda))
                w[y_hat, :] = w[y_hat, :] * (1 - (eta * lamda))
    return w

