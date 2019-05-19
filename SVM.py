import numpy as np
import random


def SVM(x_train,y_train):
    epoches = 1000
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
                if (y_hat == 0 or y == 0) and (y_hat == 1 or y == 1):
                    y_3 = 2
                if (y_hat == 0 or y == 0 ) and (y_hat == 2 or y == 2):
                    y_3 = 1
                if (y_hat == 1 or y == 1) and (y_hat == 2 or y == 2):
                    y_3 = 0
                w[y_3, :] = w[y_3, :] * (1 - (eta * lamda))

    return w

