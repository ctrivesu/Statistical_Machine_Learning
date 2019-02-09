__author__ = 'Sushant'

import os
import struct
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot
from numpy import linalg as LA

Data_Samples = 60000
Test_Samples = 10000
iteration_no = 100
label_count = 10

# READ DATA FUNCTION
def read(dataset="training", path="."):
    if dataset is "training":
        fname_img = os.path.join(path, 'train-images.idx3-ubyte')
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


# SHOW SINGLE IMAGE - takes a numpy array
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


# FUNCTIONS PROTOTYPED BY ME


# SIGMOID FUNCTION
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

# LABEL FOUND
# Label array update in comparison with given Label
def label_found(a, given_label):
    if a == given_label:
        return 1
    else:
        return 0


# LR+GD
# OPTIMIZE
def optimize(w, X, L, Xt, Lt, learning_rate, count_i):

    # print("SHAPES: ", w.shape, X.shape, L.shape, Xt.shape, Lt.shape)
    # print(w)
    cost_list = []
    m = X.shape[1]
    # NORMALIZE
    X = X/255.0
    Xt = Xt/255.0

    # Label updating to 0/1
    lf_func = np.vectorize(label_found)

    for given_label in range(label_count):
        lb_train = lf_func(L, given_label)

        # Logistic Regression + Gradient Descent code
        z = np.dot(w[given_label], X)
        p_est = sigmoid(z)
        # cost = -1.0 / (m * np.sum((L * np.log(p_est)) + ((1.0 - L) * np.log(1.0 - p_est))))
        # temp = np.sum(L * np.log(p_est) + (1.0 - L) * np.log(1.0 - p_est))
        # cost = (-1.0 / m) * (temp)
        cost = (-1.0 / Data_Samples) * (np.dot(lb_train, np.log(p_est.T) + (np.dot((1 - lb_train), np.log(1 - p_est.T)))))
        cost = cost[0]
        cost_list.append(cost)
        dw = (1.0 / Data_Samples) * np.dot(X, (p_est - lb_train).T)
        dw = dw.T

        # print("DW: ", dw)
        # print("DW: ", dw)
        # w[given_label].reshape(1, 784)
        # w -= learning_rate * dw
        wtemp = w[given_label].reshape(1, 784)
        wtemp -= learning_rate * dw
        w[given_label] = wtemp
    np.savetxt("Weights_" + str(count_i) + ".txt", w, delimiter=',')



    # FIND ACCURACY FOR EACH CLASS
    p_est_test = sigmoid(np.dot(w, Xt))
#    print("P_EST: ", p_est_test)
    y_est_test = np.argmax(p_est_test, axis=0)
    y_est_test = y_est_test.reshape((1, Test_Samples))
    count_error = 0
    for i in range(Test_Samples):
        if y_est_test[0][i] != Lt[0][i]:
            count_error += 1

    error = (float(count_error)/ Test_Samples) * 100
    accuracy = 100 - error
    print("Iteration No: ", count_i, "Accuracy: ", accuracy)
    return w

def data_reshape(L, I, dim):
    # GET X Images, X Labels from the given dataset L,I
    X = np.zeros((dim, 784))
    for i in range(dim):
        X[i][:] = I[i].reshape((1, 784))
    X = X.T
    L = L[0:dim].reshape((1, dim))
    print("Data Samples Used: ", dim)
    return (X, L)  

# MAIN FUNCTION
[L, I] = read("training")
print("Training Data Read")
[X, L] = data_reshape(L, I, Data_Samples)

[Lt, It] = read("testing")
print("Testing Data Read")
[Xt, Lt] = data_reshape(Lt, It, Test_Samples)


# show_choice = input("Show label count?(0/1)")
show_choice = "0"
# LISTING COUNT FOR EACH LABEL
if show_choice == "1":
    for i in range(10):
        print("digit", i, "appear", np.count_nonzero(L == i), "times")

# input("Press Enter to continue")
dim = X.shape[0]  # no of features

# Multi Classifier
w = np.zeros((10, dim))
for i in range(iteration_no):
    w = optimize(w, X, L, Xt, Lt, 0.1, i)





