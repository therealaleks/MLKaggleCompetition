# to extend assg dataset with resized MNIST
from keras.datasets import mnist
import numpy as np
import cv2
import h5py
import matplotlib.pyplot as plt
from matplotlib import pyplot

# load dataset
(trainX, trainy), (testX, testy) = mnist.load_data()
# summarize loaded dataset
trainX = np.concatenate((trainX,testX), axis=0)
trainy = np.concatenate((trainy,testy), axis=0)

trainX = trainX.reshape(70000,28,28)

resizedSet = []

for digit in trainX :
    digit = resized = cv2.resize(digit, (12,12), interpolation = cv2.INTER_AREA)
    resizedSet.append(digit)

f = h5py.File('data.h5', 'r')
tsd = np.array(f['data_x']).reshape(279870, 12, 12)
trl = np.array(f['data_y'])

tsd = np.concatenate((tsd,resizedSet), axis=0)
trl = np.concatenate((trl,trainy), axis=0)

h5f = h5py.File('extendedMnist.h5', 'w')
h5f.create_dataset('data_x', data=tsd)
h5f.create_dataset('data_y', data=trl)
h5f.close()