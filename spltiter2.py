import h5py
import numpy as np
import keras
from keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Dropout, Input, InputLayer
from keras.utils import to_categorical
from keras.models import Sequential, save_model, load_model
import cv2

f = h5py.File('MNIST_synthetic.h5', 'r')
tsd = np.array(f['test_dataset']).reshape(14000, 64,64).astype('float32')
tsd = np.delete(tsd, slice(26), 1)
tsd = np.delete(tsd, slice(12,38), 1)
tsd = np.delete(tsd, slice(0,2), 2)
tsd = np.delete(tsd, slice(60,62), 2)


#citeria for detected digit
threshold = 9
min_digit_width = 2
min_digit_height = 7
max_digit_width = 12
# max_digit_height = 12
min_area = 17
min_wh_ratio = 7

#total number of digits detected
numberContours = 0

xs = []
for n,digits in enumerate(tsd):
        temp_xs = []
        temp_array = []
        _, t_img = cv2.threshold(np.array(digits).astype(np.uint8), threshold, 255, cv2.THRESH_BINARY)
        #detect digits
        contours, _ = cv2.findContours(t_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #number of contours that meet the criteria for this image
        validContours = 0

        #obtain image for drawing
        img = digits.copy()
        for c in contours:
                # compute the bounding box of the contour
                (x, y, w, h) = cv2.boundingRect(c)
                #draw rectangle around detected digit
                img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 1)

                #check if detected digit meets the criteria
                if ((w*h > min_area or h/w > min_wh_ratio) and w < max_digit_width):
                        numberContours += 1
                        validContours += 1
                        #isolate digit
                        # isolate(x,y,w,h,digits)
                        temp_xs.append((x,y,w,h,))

        if validContours > 5:
            temp_xs.sort(key = lambda xx:xx[2]* xx[3])
            temp_xs.pop(0)
        temp_xs.sort(key = lambda xx:xx[0])
        for i in temp_xs:
            canvas = np.zeros(144).reshape( (12,12) )
            (x,y,w,h) = i
            z = np.take(digits, [k for k in range(x, x+w)], axis=1)
            canvas[:z.shape[0], :z.shape[1]] = z
            temp_array.append(canvas)

        for i in range(5-validContours):
            canvas = np.zeros(144).reshape( (12,12) )
            temp_array.append(canvas)
        temp_array=np.array(temp_array)
        xs.append(temp_array)


h5f = h5py.File('data_test.h5', 'w')
h5f.create_dataset('data_x', data=np.array(xs))
h5f.close()
