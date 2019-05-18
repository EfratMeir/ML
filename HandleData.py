import numpy as np
from sklearn.utils import *
import SVM
import Percpetron
import PA
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

def divideToKfoldCV(datax, datay, k):

    datax, datay = shuffle(datax, datay, random_state=1)

    ret = []
    allData = np.append(datax, np.asarray(datay, None, 'F'), axis=1)

    dataLen = len(datax)
    groupLen = round(dataLen / k)

    for i in range(k):
        tmp = np.zeros(dataLen)
        indices = range(i* groupLen, (i + 1) * groupLen - 1)
        tmp[indices] = 1

        a = 1
        tmp = (tmp == 1)

        tmpTest = allData[tmp,:]
        tmpTrain = allData[~tmp,:]

        testx = np.asarray(tmpTest[:,:-1])
        testy = np.asarray(tmpTest[:, -1])
        trainx = np.asarray(tmpTrain[:,:-1])
        trainy = np.asarray(tmpTrain[:, -1])
        ret.append([testx, testy, trainx, trainy])

    return ret

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
        predicted[i] = np.argmax(np.dot(theta, X[i]))
    return predicted


def evaluate(predicted_y, true_y):
    num_samples = len(predicted_y)
    counter = 0
    for i in range(num_samples):
        #if predicted_y[i][0] == true_y[i][0]: #if predict is good
        if predicted_y[i][0] == true_y[i]: #if predict is good
            counter = counter + 1
    precision = counter / num_samples
    return precision


def script():
    #################### flow - script: ####################

    # from file to nparray
    datax = initDataToDS('train_x.txt')
    datay = initDataToDS('train_y.txt')

    # change to correct types:
    datax = transformCategoralLabels(datax)
    datax = transformToFloat(datax)
    datay = transformToFloat(datay)

    # just for us (not for submission) - divide datax to new data set and train set:
    #testx, testy, trainx, trainy = divideNPDataToTestTrain(datax, datay, 4)

    # CV:
    k = 3
    res = divideToKfoldCV(datax, datay,k)  # res is a list with k rows, where each row is a [testx, testy, trainx, trainy] data
    avgPrecision = 0
    for i in range(k):
        [testx, testy, trainx, trainy] = res[i]

        # normalize: choose zScore or MinMax
        trainx, testx = normalizeZscore(trainx, testx)
        #trainx, testx = normalizeMinMax(trainx, testx)

        #w = SVM.SVM(trainx,trainy)
        #w = Percpetron.perceptron(trainx,trainy)
        w = PA.PA(trainx, trainy)
        predict_train = predict_y(trainx, w)
        predict_test = predict_y(testx, w)
        precision_test = evaluate(predict_test, testy)
        precision_training = evaluate(predict_train, trainy)
        print("precision on training set is " , precision_training)
        print("precision on test set is " , precision_test)

        avgPrecision += precision_test
    avgPrecision = avgPrecision / k

    print("after " + str(k) + " iteraions of cross validation, precision on test set is " , str(avgPrecision))

script()


# =======
#     testx, testy, trainx, trainy = divideNPDataToTestTrain(datax, datay, 4)
#
#     ################################################
#     import sys
#     args = sys.argv[1:]
#     testxFileName = args[2]
#     testx = initDataToDS(testxFileName)
#     testx = transformCategoralLabels(testx)
#     test2x = transformToFloat(testx)
#
#     ##################################################
#     # normalize: choose zScore or MinMax
#     # trainx, testx = normalizeZscore(trainx, testx)
#     trainx, test2x = normalizeMinMax(trainx, test2x)
#
#     w = SVM.SVM(trainx,trainy)
#     # w = Percpetron.perceptron(trainx,trainy)
#     # predict_train = predict_y(trainx, w)
#     predict_test = predict_y(test2x, w)
#     precision_test = evaluate(predict_test, testy)
#     # precision_training = evaluate(predict_train, trainy)
#     # print("precision on training set is " , precision_training)
#     print("precision on test set is " , precision_test)
#     print (", SVM: " , predict_test)
# #
# # script()
# >>>>>>> 66ed454f80438d82cbedeb33e8684de19ba2e409
