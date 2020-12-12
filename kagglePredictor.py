import h5py
import numpy as np
import keras
from keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Dropout, Input, InputLayer
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import cv2
import h5py

#citeria for detected digit
threshold = 9
min_digit_width = 2
min_digit_height = 7
max_digit_width = 12
# max_digit_height = 12
min_area = 17
min_wh_ratio = 7

# Print iterations progress
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()

def split(digits):
    xs = []

    temp_xs = []
    #apply threshold
    _, t_img = cv2.threshold(np.array(digits).astype(np.uint8), threshold, 255, cv2.THRESH_BINARY)
    # detect digits
    contours, _ = cv2.findContours(t_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # number of contours that meet the criteria for this image
    validContours = 0

    # obtain image for drawing
    img = digits.copy()
    for c in contours:
        # compute the bounding box of the contour
        (x, y, w, h) = cv2.boundingRect(c)
        # draw rectangle around detected digit
        #img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 1)

        # check if detected digit meets the criteria
        if ((w * h > min_area or h / w > min_wh_ratio) and w < max_digit_width):
            validContours += 1
            temp_xs.append((x, y, w, h,))

    temp_xs.sort(key=lambda xx: xx[0])

    if(len(temp_xs) > 5):
        temp_xs = []
        validContours = 0

    for i in temp_xs:
        canvas = np.zeros(144).reshape((12, 12))
        (x, y, w, h) = i
        z = np.take(digits, [k for k in range(x, x + w)], axis=1)
        canvas[:z.shape[0], :z.shape[1]] = z
        xs.append(canvas)
    for i in range(5 - validContours):
        canvas = np.zeros(144).reshape((12, 12))
        xs.append(canvas)

    return np.array(xs)

def predict( kaggleTestSet, model):

    kaggleTestSet = np.delete(kaggleTestSet, slice(26), 1)
    kaggleTestSet = np.delete(kaggleTestSet, slice(12, 38), 1)
    kaggleTestSet = np.delete(kaggleTestSet, slice(0, 2), 2)
    f = h5py.File('data_test.h5', 'r')
    tsd = np.array(f['data_x'])
    results = []
    #printProgressBar(0, len(kaggleTestSet), prefix = 'Progress:', suffix = 'Complete', length = 50)
    for i,digits in enumerate(tsd):
        #printProgressBar(i+1, len(kaggleTestSet), prefix = 'Progress:', suffix = 'Complete', length = 50)
        print(i)
        #splitt = split(digits)
        results.append(model.predict(digits.reshape(5,12,12,1).astype('float32')/255))
        #plt.matshow(digits)
        #plt.show()

    classes = np.argmax(np.array(results), axis=2)
    return classes


