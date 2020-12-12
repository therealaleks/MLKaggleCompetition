import h5py
import numpy as np
import keras
from keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Dropout, Input, InputLayer
from keras.utils import to_categorical
from keras.models import Sequential, save_model, load_model


f = h5py.File('MNIST_synthetic.h5', 'r')
tsd = np.array(f['test_dataset']).astype('float32')/255
trd = np.array(f['train_dataset']).astype('float32')/255
trl = to_categorical(np.array(f['train_labels']))
trl = trl.reshape(56000, 55)
trd = np.delete(trd, slice(26), 1)
trd = np.delete(trd, slice(12,38), 1)
trd = np.delete(trd, slice(0,2), 2)
trd = np.delete(trd, slice(60,62), 2)
tsd = np.delete(tsd, slice(26), 1)
tsd = np.delete(tsd, slice(12,38), 1)
tsd = np.delete(tsd, slice(0,2), 2)
tsd = np.delete(tsd, slice(60,62), 2)

h = 12
w = 60
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
model.add(Dense(55, activation='softmax'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy'])
model.fit(trd, trl, epochs=1, verbose=1)

predictions = model.predict(tsd).reshape(14000, 5, 11)
classes = np.argmax(predictions, axis=2)
with open("some", "w") as f:
	for n, i in enumerate(classes):
		f.write(f"{n},{''.join([str(j) for j in i])}\n")
