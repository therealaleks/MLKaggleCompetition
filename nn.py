import h5py
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.layers import BatchNormalization
from keras.layers import Dense, GlobalAveragePooling2D, AveragePooling2D
from keras import backend as K
from keras.layers import Input
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from numpy import mean
from numpy import std
from matplotlib import pyplot
from sklearn.model_selection import KFold
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
import keras.layers 
from keras.layers import Flatten
from keras.optimizers import SGD
import pickle
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras

f = h5py.File('/home/tony/Documents/MNIST_synthetic.h5', 'r')
tsd = np.array(f['test_dataset']).astype('float32')/255
trd = np.array(f['train_dataset']).astype('float32')/255
trl = np.array(f['train_labels'])

model = keras.Sequential()
model.add(Conv2D(32, kernel_size=(5,5), activation='relu', input_shape=(64,64,1)))
model.add(keras.layers.Dropout(0.01, noise_shape=None, seed=None))
model.add(MaxPooling2D(pool_size=(4, 4)))

model.add(Conv2D(64, kernel_size=(3,3), activation='relu')) 
model.add(keras.layers.Dropout(0.01, noise_shape=None, seed=None))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, kernel_size=(3,3), activation='relu')) 
model.add(keras.layers.Dropout(0.01, noise_shape=None, seed=None))
model.add(MaxPooling2D(pool_size=(2, 2)))

# model.add(Conv2D(64, kernel_size=(3,3), activation='relu')) 
# model.add(keras.layers.Dropout(0.01, noise_shape=None, seed=None))
# model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(5, activation='softmax'))


model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
history = model.fit(trd, trl, epochs=10, verbose=1)
