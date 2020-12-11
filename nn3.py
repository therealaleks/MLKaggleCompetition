import h5py
import numpy as np
import keras
from keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Dropout, Input, InputLayer
from keras.utils import to_categorical
from keras.models import Sequential, save_model, load_model
from kagglePredictor import predict
import matplotlib.pyplot as plt

f = h5py.File('data.h5', 'r')
trd = np.array(f['data_x']).reshape(279870, 12, 12, 1).astype('float32')/255
trl = to_categorical(np.array(f['data_y']))
print(trd.shape)
print(trl.shape)

h = 12
w = 12
d = 1

model = Sequential()
model.add(InputLayer((h, w, d)))
model.add(Conv2D(filters=16, kernel_size=(3, 3), strides=1, padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Conv2D(filters=32, kernel_size=(3, 3), strides=1, padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(128))
model.add(Dropout(0.5))
model.add(Dense(11, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
model.fit(trd,trl, epochs=400, verbose=1)

print('predicting')
f = h5py.File('MNIST_synthetic.h5', 'r')
tsd = np.array(f['test_dataset']).reshape(14000, 64,64).astype('float32')
classes = predict(tsd[:14000], model)

for n,i in enumerate(classes):
    #show 20 first results
    if n < 20:
        plt.matshow(tsd[n])
        plt.show()
    print(n,"".join([str(j) for j in i]), sep=",")
