import random
import sys
import math
import numpy
import pylab
from matplotlib.colors import ListedColormap

nClasses = 3
nItemInClass = 40
testPercent = 0.1
k = 10

def generateData (numberOfClassEl, numberOfClasses):

    data = []
    for classNum in range(numberOfClasses):

       centerX = random.random() * 5.0
       centerY = random.random() * 5.0

       for rowNum in range(numberOfClassEl):
           data.append([ [random.gauss(centerX, 0.5), random.gauss(centerY, 0.5)], classNum ])

    return data

# raw_data = generateData(5, 5)
# print(raw_data)


def showData (nClasses, nItemInClass):
    trainData = generateData(nClasses, nItemInClass)
    classColormap  = ListedColormap(['#FF0000', '#00FF00', '#FFFFFF'] )
    pylab.scatter([trainData[i][0][0] for i in range(len(trainData))],
               [trainData[i][0][1] for i in range(len(trainData))],
               c=[trainData[i][1] for i in range(len(trainData))],
               cmap=classColormap)
    pylab.show()



def splitTrainTest (data, testPercent) :
    trainData = []
    testData = []

    for row in data:
        if random.random() < testPercent:
            testData.append(row)
        else:
            trainData.append(row)
    return trainData, testData

def classifyKNN (trainData, testData, k, numberOfClasses):

    def EulerDist(a , b):
        return math.sqrt((a[0] - b[0]) ** 2 + ( a[1] - b[1]) ** 2)

    testLabels = []

    for testPoint in testData:
        testDist = [[EulerDist(testPoint, trainData[i][0]), trainData[i][1]] for i in range(len(trainData))]

        stat = [0 for i in range(numberOfClasses)]


        for d in sorted(testDist)[0:k]:
            stat[d[1]] += 1

        testLabels.append( sorted( zip( stat, range(numberOfClasses)), reverse=True)[0][1] )

    return testLabels

def calculateAccuracy (nClasses, nItemInClass, k, testPercent):
    data = generateData(nItemInClass, nClasses)
    trainData, testDataWithLabels = splitTrainTest(data, testPercent)
    testData = [testDataWithLabels[i][0] for i in range(len(testDataWithLabels))]

    testDataLabels = classifyKNN(trainData, testData, k, nClasses)
    print ("Accuracy: ", sum([int(testDataLabels[i]==testDataWithLabels[i][1]) for i in range(len(testDataWithLabels))]) / float(len(testDataWithLabels)))

#Visualize classification regions
def showDataOnMesh (nClasses, nItemsInClass, k):
    #Generate a mesh of nodes that covers all train cases
    def generateTestMesh (trainData):
        x_min = min( [trainData[i][0][0] for i in range(len(trainData))] ) - 1.0
        x_max = max( [trainData[i][0][0] for i in range(len(trainData))] ) + 1.0
        y_min = min( [trainData[i][0][1] for i in range(len(trainData))] ) - 1.0
        y_max = max( [trainData[i][0][1] for i in range(len(trainData))] ) + 1.0
        h = 0.05
        testX, testY = numpy.meshgrid(numpy.arange(x_min, x_max, h),
                                   numpy.arange(y_min, y_max, h))
        return [testX, testY]
    trainData      = generateData (nItemsInClass, nClasses)
    testMesh       = generateTestMesh (trainData)
    testMeshLabels = classifyKNN (trainData, zip(testMesh[0].ravel(), testMesh[1].ravel()), k, nClasses)
    classColormap  = ListedColormap(['#FF0000', '#00FF00', '#FFFFFF'])
    testColormap   = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAAA'])
    pylab.pcolormesh(testMesh[0],
                  testMesh[1],
                  numpy.asarray(testMeshLabels).reshape(testMesh[0].shape),
                  cmap=testColormap)
    pylab.scatter([trainData[i][0][0] for i in range(len(trainData))],
               [trainData[i][0][1] for i in range(len(trainData))],
               c=[trainData[i][1] for i in range(len(trainData))],
               cmap=classColormap)
    pylab.show()


data = generateData(40,3)

trainData, testDataWithLabels = splitTrainTest(data, testPercent)
testData = [testDataWithLabels[i][0] for i in range(len(testDataWithLabels))]

print("Raw data: ")
print(data)
print("\n")

print("Train Data: ")
print(trainData)

print("Test Data with Labels")
print(testDataWithLabels)

print("Test Data: ")
print(testData)

classifiedData = classifyKNN(trainData, testData, 10, 3)

print(classifiedData)

# calculateAccuracy(nClasses, nItemInClass, k, testPercent)

print(showDataOnMesh(nClasses, nItemInClass, k))