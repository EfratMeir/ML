import numpy as np
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

datax = initDataToDS('train_x.txt')
datay = initDataToDS('train_y.txt')


# add x and y together:
numFeatures = len(datax[0])
alldata = np.zeros((len(datax[:,1]), numFeatures + 1))
alldata = np.append(datax, datay, axis=1)
alldata = np.random.permutation(alldata)

datax = alldata[:,:-1]
datay = alldata[:,-1]

testx= []
testy = []
trainx = []
trainy = []

k = 4
dataLen = len(datax)
groupLen = round(dataLen / k)

for i in range(groupLen):
    testx.append(datax[i])
    testy.append(datay[i])

testx = np.asarray(testx)
testy = np.asarray(testy,None, 'F')

for i in range(groupLen, dataLen):
    trainx.append(datax[i])
    trainy.append(datay[i])

trainx = np.asarray(trainx)
trainy = np.asarray(trainy,None,'F')


w, y_hat = Percpetron.perceptron(trainx, trainy)
print("dd")