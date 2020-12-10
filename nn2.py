import h5py
import numpy as np
import keras
from keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Dropout, Input, InputLayer
from keras.utils import to_categorical


f = h5py.File('/home/tony/Documents/MNIST_synthetic.h5', 'r')
tsd = np.array(f['test_dataset']).astype('float32')/255
trd = np.array(f['train_dataset']).astype('float32')/255
trl = to_categorical(np.array(f['train_labels']))
trl = trl.reshape(56000, 55)
trd = np.delete(trd, slice(26), 1)
trd = np.delete(trd, slice(12,38), 1)
trd = np.delete(trd, slice(0,2), 2)
trd = np.delete(trd, slice(60,62), 2)
print(trl[5])

h = 12
w = 60
d = 1

cnn_model = keras.models.Sequential()
cnn_model.add(InputLayer((h, w, d)))
cnn_model.add(Conv2D(filters=16, kernel_size=(3, 3), strides=1, padding='same', activation='relu'))
cnn_model.add(MaxPool2D(pool_size=(2, 2)))
cnn_model.add(Dropout(0.2))
cnn_model.add(Conv2D(filters=32, kernel_size=(3, 3), strides=1, padding='same', activation='relu'))
cnn_model.add(MaxPool2D(pool_size=(2, 2)))
cnn_model.add(Dropout(0.2))
cnn_model.add(Flatten())
cnn_model.add(Dense(128))
cnn_model.add(Dropout(0.5))
cnn_model.add(Dense(55, activation='softmax'))

cnn_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy'])


history = cnn_model.fit(trd, trl, epochs=3, verbose=1)
