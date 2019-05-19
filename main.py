import sys
from HandleData import *
import SVM, Percpetron, PA

# putt the script here
def run():
    args = sys.argv[1:]
    trainxFileName = args[0]
    trainyFileName = args[1]
    testxFileName = args[2]

    # from file to nparray
    trainx = initDataToDS(trainxFileName)
    trainy = initDataToDS(trainyFileName)
    testx = initDataToDS(testxFileName)

    # change to correct types:
    trainx = transformCategoralLabels(trainx)
    testx = transformCategoralLabels(testx)

    trainx = transformToFloat(trainx)
    trainy = transformToFloat(trainy)
    testx = transformToFloat(testx)

    trainx, trainy = shuffle2arr(trainx, trainy)
    # trainx, trainy = shuffle(trainx, trainy, random_state=1)

    # normalize: choose zScore or MinMax ****normlize without the category column*******
    # trainx, testx = normalizeZscore(trainx, testx)
    trainx, testx = normalizeMinMax(trainx, testx)

    # normalize: choose zScore or MinMax
    # trainx, testx = normalizeZscore(trainx, testx)
    # trainx, testx = normalizeMinMax(trainx, testx)

    # train the models:
    percpetron_w = Percpetron.perceptron(trainx, trainy)
    svm_w = SVM.SVM(trainx,trainy)
    pa_w = PA.PA(trainx, trainy)

    # predict labels:
    percpetron_predict = predict_y(testx, percpetron_w)
    svm_predict = predict_y(testx, svm_w)
    pa_predict = predict_y(testx, pa_w)

    for i in range(len(testx)):
        print("perceptron: " + str(percpetron_predict[i][0].astype(np.int)) + ", svm: " + str(svm_predict[i][0].astype(np.int)) + ", pa: " + str(pa_predict[i][0].astype(np.int)))


run()
