import numpy as np
import random

def perceptron(x_train,y_train):
    epoches = 100
    class_num = 3
    features_num = len(x_train[0])

    w = np.zeros((class_num, features_num))
    eta = 0.1
    for e in range(epoches):
        for x, y in zip(x_train, y_train):
            y = int(y)
            # predict
            y_hat = np.argmax(np.dot(w, x))
            #update
            if y != y_hat:
                w[y, :] = w[y, :] + eta * x
                w[y_hat, :] = w[y_hat, :] - eta * x
    return w

