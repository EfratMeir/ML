import sys
import HandleData


def createTestFile(dataxFile, datayFile, k):
    allDatax = HandleData.initDataToDS(dataxFile)
    allDatay = HandleData.initDataToDS(datayFile)

args = sys.argv[1:]
print(args)