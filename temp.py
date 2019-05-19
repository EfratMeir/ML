import sys
from HandleData import *
def script():
    args = sys.argv[1:]
    trainx = initDataToDS('train_x.txt')
    trainy = initDataToDS('train_y.txt')
    testxFileName = args[2]
    testx = initDataToDS(testxFileName)
    # trainx = initDataToDS(trainxFileName)
    # trainy = initDataToDS(trainyFileName)
    trainx = transformCategoralLabels(trainx)
    testx = transformCategoralLabels(testx)


    trainx = transformToFloat(trainx)
    trainy = transformToFloat(trainy)
    test2x = transformToFloat(testx)
    trainx, trainy = shuffle2arr(trainx, trainy)

    trainx, test2x = normalizeMinMax(trainx, test2x)
    w = SVM.SVM(trainx, trainy)
    predict_test = predict_y(test2x, w)
    print(predict_test)
script()