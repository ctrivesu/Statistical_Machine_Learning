import csv
import random
import math
import operator

import os
import struct
import numpy as np
from numpy import linalg as LA
import matplotlib as mpl
from matplotlib import pyplot
from collections import Counter

_author_ = 'Sushant Trivedi'
Data_Samples = 1000
Test_Samples = 500
Count_to_Write = 15
k_arr = [100, 90, 80, 70, 50, 30, 10, 5, 3, 1]
count_error = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# k_arr = [1, 3, 5, 10, 30, 50, 70, 80, 90, 100]

# READ FUNCTION
def read(dataset="training", path="."):
    if dataset is "training":
        fname_img = os.path.join(path, 'train-images.idx3-ubyte')  # "./train-images.idx3-ubyte"
        fname_lbl = os.path.join(path, 'train-labels.idx1-ubyte')
    elif dataset is "testing":
        fname_img = os.path.join(path, 't10k-images.idx3-ubyte')
        fname_lbl = os.path.join(path, 't10k-labels.idx1-ubyte')
    else:
        raise ValueError("dataset must be 'testing' or 'training'")

    # Load everything in some numpy arrays
    with open(fname_lbl, 'rb') as flbl:
        _, _ = struct.unpack(">II", flbl.read(8))
        lbl = np.fromfile(flbl, dtype=np.int8)

    with open(fname_img, 'rb') as fimg:
        _, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        img = np.fromfile(fimg, dtype=np.uint8).reshape(len(lbl), rows, cols)

    return (lbl, img)


# SHOW FUNCTION
def show(image):
    """
    Render a given numpy.uint8 2D array of pixel data.
    """
    fig = pyplot.figure()
    ax = fig.add_subplot(1, 1, 1)
    imgplot = ax.imshow(image, cmap=mpl.cm.Greys)
    imgplot.set_interpolation('nearest')
    ax.xaxis.set_ticks_position('top')
    ax.yaxis.set_ticks_position('left')
    pyplot.show()

# ACCURACY FOR K ELEMENTS
def knn(trainingSet, testInstance, correct_label):
    dists = []

    # Calculate distances for the test point
    for x in trainingSet:
        distance = LA.norm((x[0:784] - testInstance[0:784]))

        dists.append((distance, x[-1]))  # CHECK HERE - Does x[-1] access labels?

    dists.sort(key=operator.itemgetter(0))

    # print("DIST: ", dists)
    # print(dists[-1][-1])

    # CREATE A LIST WITH K LABELS
    sorted_labels = []
    for x in dists:
        sorted_labels.append(x[1])

    # print("LABELS: ", sorted_labels)
    # CALCULATIONS FOR ALL K
    t = 0
    for k in k_arr:
        y_est, most_count = Counter(sorted_labels[0:k]).most_common(1)[0]  # GETS THE VOTES HERE
        # print("K: ", k, "Y_EST: ", y_est, "TRUE LABEL: ", correct_label)
        if y_est != correct_label:
            count_error[t] += 1
        t += 1

# MAIN BODY START
# TRAINING DATA SET READ
(L_train, I_train) = read()
L_train = L_train.reshape(60000, 1)
temp_T = I_train.flatten().reshape(60000, 784)
temp_T = temp_T / 255.0

# TRAINING SET SIZE SELECTION
temp_T = temp_T[0:Data_Samples]
L_train = L_train[0:Data_Samples]
trainingSet = np.hstack((temp_T, L_train))
print('Training DataSet Ready')

# TEST DATA SET READ
(L_test, I_test) = read("testing")
L_test = L_test.reshape(10000, 1)
temp_E = I_test.flatten().reshape(10000, 784)
temp_E = temp_E / 255.0

# TEST SET SIZE SELECTION
temp_E = temp_E[0:Test_Samples]
L_test = L_test[0:Test_Samples]
testSet = np.hstack((temp_E, L_test))
print('Testing DataSet Ready')

# PREDICTIONS

for x in range(len(testSet)):
    knn(trainingSet, testSet[x], testSet[x][-1])

    # STORE IN BETWEEN RESULTS
    if x == Count_to_Write:
        fp = open("Accuracy_" + str(x) + ".txt", "w+")
        fp.write(" ".join(str(x) for x in count_error))
        fp.close()
    # print("Sample", x)

# FINAL ERROR COUNT FILE STORE
fp = open("Accuracy_FINAL.txt", "w+")
fp.write(" ".join(str(x) for x in count_error))
fp.close()

# CALCULATE ACCURACY
print("Error_Count: ", count_error)
for i in range(len(k_arr)):
    accuracy = (count_error[i] / float(Test_Samples)) * 100.0
    accuracy = 100 - accuracy
    print("K: ", k_arr[i], " Accuracy: ", accuracy)

print("KNN Completed")











