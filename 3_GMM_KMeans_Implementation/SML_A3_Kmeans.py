__author__ = 'Sushant'

import csv
import random

import numpy as np
from numpy import linalg as LA
from matplotlib import pyplot


# READING THE FILE DATA
with open(r"C:\Users\Sushant\Desktop\ASU Courses\CSE575 - Statistical Machine Learning\Assignment - 3\audioData.csv", 'r') \
        as fp:
    data_iter = csv.reader(fp, delimiter=',', quotechar='"')
    data = [data for data in data_iter]

data_array = np.asarray(data)
data_array = data_array.astype(np.float)
print(data_array)
Kmeans = np.zeros(10)
print("KMeans")

# NORMALIZE THE ARRAY
# f_mean = np.sum(data_array, axis=0) / 128
# print("MEANS: ", f_mean)
# f_max = np.amax(data_array, axis=0)
# f_min = np.amin(data_array, axis=0)
# data_array = (data_array - f_min) / (f_max - f_min)
# data_norm = data_array - f_mean

# K-Means Algorithm
def Kmeans(data, k):

    delta = 999999
    K = np.zeros((10, 13))
    cond = np.zeros((10, 1))

    while (K[0:k].size - np.count_nonzero(K[0:k])) != 0:
        # print("Restart KMeans: ", K[0:k].size, np.count_nonzero(K[0:k]))

        # Initializing the means
        np.random.seed(10)
        for i in range(k):
            K[i] = data[random.randint(0, 10)]

        k_count = np.zeros((10, 1))
        k_count_old = np.random.rand(10, 1)

        # Kmeans Algorithm
        while not np.array_equal(k_count_old, k_count):
            k_count_old = k_count
            k_sum = np.zeros((10, 13))
            temp = np.zeros((10, 13))
            k_count = np.zeros((10, 1))
            cost = 0
            for i in range(data.shape[0]):   # Run it for all points
                temp[0:k] = K[0:k] - np.tile(data.astype('float')[i], (k, 1))    # Distance from each mean
                t = LA.norm(temp, axis=1)
                k_index = np.argmin(t[0:k])
                cost += np.power(t[k_index], 2)
                k_sum[k_index] += data[i].astype('float')
                k_count[k_index] += 1
            K = np.divide(k_sum, k_count, out=np.zeros_like(k_sum), where=k_count != 0)
        k_count[0:k].sort(axis=0)
        # print("Final K_Count: ", k_count[0:k])
        cond = LA.norm(K[0:k], axis=1)
        # print(cond)
    print("K: ", k, "COST: ", cost)
    return cost

# MAIN FUNCTION
temp = []
for i in range(2, 11):
    temp.append(Kmeans(data_array, i))
print("TEMP: ", temp)
pyplot.plot([i for i in range(2, 11)], temp)
pyplot.show()