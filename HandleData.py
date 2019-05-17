import numpy as np
from sklearn.utils import *
from sklearn import preprocessing


def normalizeZscore(trainx, testx):
    mean = np.mean(trainx, axis=0)
    std = np.std(trainx, axis=0)
    trainx = getZscore(trainx, mean, std)
    testx = getZscore(testx, mean, std)
    return trainx, testx


def getZscore(x, mean, std):
    return (x - mean) / std

def normalizeMinMax(trainx, testx):
    minVal = np.min(trainx, axis=0)
    maxVal = np.max(trainx, axis=0)
    trainx = getMinMaxScore(trainx, minVal, maxVal)
    testx = getMinMaxScore(testx, minVal, maxVal)
    return trainx, testx

def getMinMaxScore(val, min, max):
    return (val - min) / (max - min)
import Percpetron
from sklearn import preprocessing


def initDataToDS(dataFile):
    filex = open(dataFile, 'r')
    datatmp = filex.read().split('\n')
    data = []
    for line in datatmp:
      data.append(line.split(','))
    data = np.asarray(data)
    return data


def transformCategoralLabels(datax):
    le = preprocessing.LabelEncoder()
    le.fit(["M", "F", "I"])
    for i in range(len(datax)):
        datax[i][0] = le.transform([datax[i][0]])[0]
    return datax


def transformToFloat(data):
    return data.astype(np.float)


def divideNPDataToTestTrain(datax, datay, k):

    datax, datay = shuffle(datax, datay, random_state=1)

    testx= []
    testy = []
    trainx = []
    trainy = []

    dataLen = len(datax)
    groupLen = round(dataLen / k)

    for i in range(groupLen):
        testx.append(datax[i])
        testy.append(datay[i])

    testx = np.asarray(testx)
    testy = np.asarray(testy,None,'F')

    for i in range(groupLen, dataLen):
        trainx.append(datax[i])
        trainy.append((datay[i]))

    trainx = np.asarray(trainx)
    trainy = np.asarray(trainy,None,'F')

    return testx, testy, trainx, trainy



def predict_y(X, theta):
#      fill code to predict the labels
    num_samples = len(X)
    predicted = np.zeros((num_samples,1))

    for i in range(num_samples):
        for x in X:
            condition = np.argmax(np.dot(theta, x))
            # if condition > 0:             WHAT TO DOOOOOOOOO
            #     predicted[i] = 1
            # else:
            #     predicted[i] = -1
    return predicted

#################### flow - script: ####################

# from file to nparray
datax = initDataToDS('train_x.txt')
datay = initDataToDS('train_y.txt')

# change to correct types:
datax = transformCategoralLabels(datax)
datax = transformToFloat(datax)
datay = transformToFloat(datay)

# just for us (not for submission) - divide datax to new data set and train set:
testx, testy, trainx, trainy = divideNPDataToTestTrain(datax, datay, 4)

# normalize: choose zScore or MinMax
#trainx, testx = normalizeZscore(trainx, testx)
trainx, testx = normalizeMinMax(trainx, testx)

w = Percpetron.perceptron(trainx,trainy)
predict = predict_y(trainx, w)
print("w is: " ,w)

