
import numpy as np

def to_centre(mat):
	x = 0
	y = 0
	cnt = 0
	for i in range(mat.shape[0]):
		for j in range(mat.shape[1]):
			if (mat[i][j] != 0):
				x += i
				y += j
				cnt += 1
	offset_x = int(x / cnt) - int(mat.shape[0] / 2)
	offset_y = int(y / cnt) - int(mat.shape[1] / 2)
	ret = np.empty((mat.shape[0], mat.shape[1]), np.uint8)
	for i in range(mat.shape[0]):
		for j in range(mat.shape[1]):
			x = i + offset_x
			y = j + offset_y
			if(x >= 0 and x < mat.shape[0] and y >= 0 and y < mat.shape[1]):
				ret[i][j] = mat[x][y]
			else:
				ret[i][j] = 0
	return ret
 
def KNN(inX, dataSet, labels, k, iNormNum):
    subtractMat = np.ones([dataSet.shape[0], 1], dtype=np.int32)*inX - dataSet
    distances = (subtractMat**iNormNum).sum(axis=1)
    sortedDistIndicies = distances.argsort()
    classCount={}          
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    sortedClassCount = sorted(classCount.items(), key=lambda s: s[1], reverse=True)
    return sortedClassCount[0][0]
#    return labels[sortedClassCount[0][0]]
     
if __name__ == '__main__':

    data_num1 = 60000 #The number of figures
    data_num2 = 10000
    fig_w = 45       #width of each figure

    trainDataSet  = np.fromfile("mnist_train_data",dtype=np.uint8)
    trainLabels = np.fromfile("mnist_train_label",dtype=np.uint8)
    testDataSet = np.fromfile("mnist_test_data",dtype=np.uint8)
    testLabels = np.fromfile("mnist_test_label",dtype=np.uint8)

    trainDataSet = trainDataSet.reshape(data_num1,fig_w,fig_w)
    testDataSet = testDataSet.reshape(data_num2,fig_w,fig_w)

    for i in range(data_num1):
        trainDataSet[i] = to_centre(trainDataSet[i])
    for i in range(data_num2):
        testDataSet[i] = to_centre(testDataSet[i])

    #reshape the matrix
    trainDataSet = trainDataSet.reshape(data_num1,fig_w*fig_w)
    testDataSet = testDataSet.reshape(data_num2,fig_w*fig_w)
    #testDataSet = testDataSet[:100]

    #data_num2=100

    iErrorNum      = 0

    for iTestInd in range(data_num2):
        KNNResult = KNN(testDataSet[iTestInd].reshape([1,45*45]), trainDataSet, trainLabels, 5, 2)
        if (KNNResult != testLabels[iTestInd]): iErrorNum += 1.0
        print("process:%d/%d_totalErrorNum:%d predict_label: %d, real_label: %d" % (iTestInd, data_num2, iErrorNum, KNNResult, testLabels[iTestInd]))
    print("\nthe total number of errors is: %d" % iErrorNum)
    print("\nthe total error rate is: %f" % (iErrorNum/float(data_num2)))
    print("\naccuracy : %f" % (1-(iErrorNum/float(data_num2))))
     
