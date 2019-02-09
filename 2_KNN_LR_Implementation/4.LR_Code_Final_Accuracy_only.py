__author__ = 'Sushant'

import os
import struct
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot

Data_Samples = 30000
Test_Samples = 10000



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


# //////////////////COMMENTED CODE////////////////
# mylist = list(set(L))
# for i in range(5):
# print(L[i])
# show(I[i])

# for i in range(3):
#     show(temp[i].reshape((28, 28)))


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
def optimize(w, X, L, iteration_no, learning_rate, given_label):
    cost_list = []
    m = X.shape[1]
    
    #NORMALIZE - 
    X = X/255.0
    
    # Label updating to 0/1
    lf_func = np.vectorize(label_found)
    L = lf_func(L, given_label)
    uu = np.ones((1, Data_Samples))
    # Logistic Regression + Gradient Descent code
    for i in range(iteration_no):
        z = np.dot(w.T, X)
        p_est = sigmoid(z)
        # cost = -1.0 / (m * np.sum((L * np.log(p_est)) + ((1.0 - L) * np.log(1.0 - p_est))))
        #temp = np.sum(L * np.log(p_est) + (1.0 - L) * np.log(1.0 - p_est))
        #cost = (-1.0 / m) * (temp)
        #DOUBT
        cost = (-1.0 / m) * ((np.dot(L,np.log(p_est.T))+(np.dot((uu-L),np.log(uu-(p_est.T)))))) 
        cost = cost[0][0]
        cost_list.append(cost)
        dw = 1.0 / m * np.dot(X, (p_est - L).T)
        w -= learning_rate * dw
        #print(i + 1, " Cost: ", cost)
    #pyplot.plot(cost_list)
    #pyplot.show()
    return w

#PREDICT
# Predict the Label - 0/1
# w - Weight Matrix
# X - Testing Data
def predict(w, X):
    m = X.shape[1]           #Testing set sample size
    z = np.dot(w.T, X)  
    p_est = sigmoid(z)
    y_prediction = np.zeros((1,m)) # Estimated Label
        
    for i in range(p_est.shape[1]):
        if (p_est[:,i] > 0.5): 
            y_prediction[:, i] = 1
        elif (p_est[:,i] <= 0.5):
            y_prediction[:, i] = 0
    return y_prediction

#Accuracy for Training Set 
def accuracy(w,X,L):
    print(w.shape, X.shape, L.shape)
    temp = w.T
    for label in range(9):
        w = temp[label].T
        w = w.reshape((784, 1))
        y_predict_train = predict(w, X)
        # Label updating to 0/1
        lf_func = np.vectorize(label_found)
        L = lf_func(L, label)
        accuracy = 100.0 - np.mean(np.abs(y_predict_train - L)*100.0)
        print(label, " Accuracy: ", accuracy)
        
        
def accuracy_all_labels(w, X, L):
    m = X.shape[1] # Samples count
    p_est = sigmoid(np.dot(w.T, X))
    #print("Size of p_est: ",p_est.shape)
    np.savetxt(r"C:\Users\strived6\Desktop\SML\P_Est_all_labels.csv", p_est, delimiter = ',') # , fmt='%.3e')
    y_est = np.argmax(p_est, axis=0)
    y_est = y_est.reshape((1,m))
    #print("Size of yest: ",y_est.shape)
    
    error = L - y_est
    bin = np.where(error > 0, 1, 0)
    error_count = np.count_nonzero(bin == 1)
    accuracy = (error_count/m)*100
    accuracy = 100 - accuracy    
    
    
    print("Accuracy: ", accuracy)
    #print(L)
    #np.savetxt(r"C:\Users\strived6\Desktop\SML\L.csv", L, delimiter = ',') # , fmt='%.3e')
    #print(y_est)
    #np.savetxt(r"C:\Users\strived6\Desktop\SML\y_est.csv", y_est, delimiter = ',') # , fmt='%.3e')

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
[L, I] = read("training", r"C:\Users\strived6\Desktop\SML")
print("Training Data Read")
[X, L] = data_reshape(L, I, Data_Samples)
# GET X Images, X Labels from the given dataset L,I
# X = np.zeros((1, 784))
# for i in range(Data_Samples):
#     X = np.concatenate((X, I[i].reshape((1, 784))), axis=0)
# X = np.delete(X, 0, 0)
# X = X.T
# print(X.shape, L[0:Data_Samples].reshape((Data_Samples, 1)).shape)
# L = L[0:Data_Samples].reshape((1, Data_Samples))
# print("Data Samples Used: ", X.shape[1])
# show_choice = input("Show label count?(0/1)")
show_choice = "0"
# LISTING COUNT FOR EACH LABEL
if show_choice == "1":
    for i in range(10):
        print("digit", i, "appear", np.count_nonzero(L == i), "times")

# input("Press Enter to continue")
dim = X.shape[0]  # no of features

# Single Classifier
# w = np.zeros((dim, 1))
# w = optimize(w, X, L, 100, 0.0001, 9)

# Multi Classifier
w = np.zeros((dim, 1))
for i in range(10):
    ret = optimize(np.zeros((dim, 1)), X, L, 100, 0.0001, i)
    w = np.concatenate((w, ret), axis=1)
    print("Iteration for label ",i, "completed")
w = np.delete(w, 0, 1)
print("W_multiclass: ", w.shape)

# Store the weight matrix in a file
np.savetxt(r"C:\Users\strived6\Desktop\SML\Weight.csv", w, delimiter = ',') # , fmt='%.3e')

print("-----TRAINING DATA-----")
accuracy_all_labels(w,X,L)
print("-----TESTING DATA-----")
[L, I] = read("testing", r"C:\Users\strived6\Desktop\SML")
print("Testing Data Read")
[X, L] = data_reshape(L, I, Test_Samples)
accuracy_all_labels(w,X,L)






