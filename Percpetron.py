import numpy as np
import random

def perceptron(x_train,y_train):
    epoches = 10
    w = random.uniform(0, 1)
    eta = 0.1
    for e in range(epoches):
        for x, y in zip(x_train, y_train):
            # predict
            y_hat = np.argmax(np.dot(w, x))
            #update
            if y != y_hat:
                w[y] = w[y, :] + eta * x
                w[y_hat, :] = w[y_hat, :] - eta * x
    return w,y_hat

