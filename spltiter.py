import h5py
import numpy as np
import keras
from keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Dropout, Input, InputLayer
from keras.utils import to_categorical
# import matplotlib.pyplot as plt
import cv2
import h5py

def isolate(x, y, w, h, img):
	#isolate digit from image
	pass


f = h5py.File('MNIST_synthetic.h5', 'r')
tsd = np.array(f['test_dataset']).reshape(14000, 64,64).astype('float32')
trd = np.array(f['train_dataset']).reshape(56000, 64,64).astype('float32')
trl = np.array(f['train_labels'])
trd = np.delete(trd, slice(26), 1)
trd = np.delete(trd, slice(12,38), 1)
trd = np.delete(trd, slice(0,2), 2)
trd = np.delete(trd, slice(60,62), 2)
tsd = np.delete(tsd, slice(26), 1)
tsd = np.delete(tsd, slice(12,38), 1)
tsd = np.delete(tsd, slice(0,2), 2)
tsd = np.delete(tsd, slice(60,62), 2)

#true number of digits per image
labelDigitCounts = []
#total number of digits
totalcount = 0

for label in trl:
	numberDigits = 0
	for digitLabel in label:
		if digitLabel != 10:
			totalcount += 1
			numberDigits += 1
	labelDigitCounts.append(numberDigits)

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
#number of failed images
fails = 0

ys = []
xs = []
for n,digits in enumerate(trd):
	temp_xs = []
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

	#if number of digits detected is not equal to the true number of digits in image
	if( validContours != labelDigitCounts[n] ):
		#increment fail count
		fails += 1
		#show image with drawn rectangles
		# plt.matshow(img)
		# plt.show()
	else:
		for number in trl[n]:
			ys.append(number)
		temp_xs.sort(key = lambda xx:xx[0])
		for i in temp_xs:
			canvas = np.zeros(144).reshape( (12,12) )
			(x,y,w,h) = i
			z = np.take(digits, [k for k in range(x, x+w)], axis=1)
			canvas[:z.shape[0], :z.shape[1]] = z
			xs.append(canvas)
		for i in range(5-validContours):
			canvas = np.zeros(144).reshape( (12,12) )
			xs.append(canvas)

xs = np.array(xs)
ys = np.array(ys)

h5f = h5py.File('data.h5', 'w')
h5f.create_dataset('data_x', data=xs)
h5f.create_dataset('data_y', data=ys)
h5f.close()

print('total number of images:' + str(len(trl)))
print('failed images:' + str(fails))
print('total number of digits:' + str(totalcount))
print('number of digits successfully detected:' + str(numberContours))
