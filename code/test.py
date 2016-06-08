# import the necessary packages
import argparse
import datetime
import imutils
import time
import cv2
import numpy as np
from matplotlib import pyplot as plt

letters = ["a","i","u","e","o","ka","ki","ku","ke","ko","sa","shi","su","se","so","fu","ha","hi","ho","he","ma","mi","mu","me","mo","n","na","ni","no","nu","ne","ra","ri","ru","re","ro","ta","chi","to","te","tsu","wa","wo","ya","yo","yu"]
lettersN = range(46)
SZ=50
bin_n = 16 # Number of bins
affine_flags = cv2.WARP_INVERSE_MAP|cv2.INTER_LINEAR

SVM_GAMMA = 5.383
SVM_C = 2.67

def deskew(img):
	m = cv2.moments(img)
	if abs(m['mu02']) < 1e-2:
		return img.copy()
	skew = m['mu11']/m['mu02']
	M = np.float32([[1, skew, -0.5*SZ*skew], [0, 1, 0]])
	img = cv2.warpAffine(img,M,(SZ, SZ),flags=affine_flags)
	return img

def hog(img):
	gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
	gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
	mag, ang = cv2.cartToPolar(gx, gy)

	# quantizing binvalues in (0...16)
	bins = np.int32(bin_n*ang/(2*np.pi))

	x = 25
	# Divide to 4 sub-squares
	bin_cells = bins[:x,:x], bins[x:,:x], bins[:x,x:], bins[x:,x:]
	mag_cells = mag[:x,:x], mag[x:,:x], mag[:x,x:], mag[x:,x:]
	hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
	hist = np.hstack(hists)
	return hist

def printNthInList(list, n):
	text = ""
	for i in range(2500):
		if i % 50 > 0:
			if list[n][i] != 0:
				text += "+"
			else:
				text += " "
		else:
			text += "\n"
	print text

# CREATE NxNpx data from picture
def make_unified_data(N):
	for letter in letters:
		for i in range(10):
			image = cv2.imread("../data/"+letter+"/"+str(i)+".png")
			image = cv2.resize(image, (N,N))
			cv2.imwrite("../data/"+letter+"/"+str(i)+"_"+str(N)+"x"+str(N)+".png", image)

def make_usable_data(x, dataN, offset):
	onePicPx = len(x[0]) * len(x[0][0])
	# Make each pictures to 1-dim array (train data) 8 picture per letter
	offset = offset
	data = []
	for i in range(len(x)/10):
		for i in range(dataN):
			data.append(x[offset+i])
		offset += 10

	data = np.array(data)
	data = data.reshape(-1,onePicPx).astype(np.float32)
	return data

# Load in the letters
def generate_image_data():
	cells = []
	for letter in letters:
		for i in range(10):
			if letter == "sa" and i == 6:
				image = cv2.imread("../data/"+letter+"/"+str(i)+"_50x50.png",0)
				thresh = cv2.threshold(image,100,255,cv2.THRESH_BINARY_INV)[1]
				thresh = cv2.dilate(thresh, np.ones((3,3),np.uint8), iterations=2)
				cells.append(thresh)
			else:
				image = cv2.imread("../data/"+letter+"/"+str(i)+"_50x50.png",0)
				thresh = cv2.threshold(image,100,255,cv2.THRESH_BINARY_INV)[1]
				thresh = cv2.dilate(thresh, np.ones((3,3),np.uint8), iterations=2)
				cells.append(thresh)

	# Make images to numpy array
	x = np.array(cells)
	deskewed = [map(deskew,row) for row in x]
	hogdata = [map(hog,row) for row in deskewed]
	return x,hogdata


######################################################
# SVM 
######################################################
def test_SVM_accuracy(x, trainN, testN, name):

	## TRAINING ###
	# Make each pictures to 1-dim array (train data) trainN picture per letter
	train = make_usable_data(x, trainN, 0)

	# Generate integer values for letters
	train_labels = np.repeat(lettersN, trainN)[:,np.newaxis]

	# Make svm 
	svm = cv2.ml.SVM_create()
	svm.setGamma(SVM_GAMMA)
	svm.setC(SVM_C)
	svm.setKernel(cv2.ml.SVM_LINEAR)
	svm.setType(cv2.ml.SVM_C_SVC)

	ok = svm.train(train,cv2.ml.ROW_SAMPLE,train_labels)

	### TESTING ###
	# Make each pictures to 1-dim array (test data) testN pictures per letter
	test = make_usable_data(x, testN, trainN)
	# Generate integer values for letters
	test_labels = np.repeat(lettersN, testN)[:,np.newaxis]

	result = svm.predict(test)

	### CHECK ACCURACY ###
	mask = result[1]==test_labels
	correct = np.count_nonzero(mask)
	accuracy = correct*100.0/result[1].size
	print name + str(accuracy)

######################################################
# SVM
######################################################



######################################################
# k-Nearest Neighbour - with picture
# x = Array of characters in format of [[px,px,px],[px,px,px],[px,px,px]] - 3x3px image.
# 
######################################################
def test_kNN_accuracy(x, trainN, testN, name):

	## TRAINING ###
	# Make each pictures to 1-dim array (train data) trainN picture per letter
	train = make_usable_data(x, trainN, 0)

	# Generate integer values for letters
	train_labels = np.repeat(lettersN, trainN)[:,np.newaxis]

	# Do the real k-nearest neighbour search
	knn = cv2.ml.KNearest_create()
	knn.train(train, cv2.ml.ROW_SAMPLE, train_labels)

	### TESTING ###
	# Make each pictures to 1-dim array (test data) testN pictures per letter
	test = make_usable_data(x, testN, trainN)
	ret,result,neighbours,dist = knn.findNearest(test,k=4)
	test_labels = np.repeat(lettersN, testN)[:,np.newaxis]

	### CHECK ACCURACY ###
	matches = result==test_labels
	correct = np.count_nonzero(matches)

	accuracy = correct*100.0/result.size
	print name + str(accuracy)

######################################################
# k-Nearest Neighbour - with picture
######################################################

# Merges the pari of rectangles into one
def pair_pairs(pairs, rects):
	for pair in pairs:
		upper = None
		lower = None
		if pair[0][1] > pair[1][1]:
			upper = pair[1]
			lower = pair[0]
		else:
			upper = pair[0]
			lower = pair[1]
		x = min(upper[0], lower[0])
		y = upper[1]
		w = abs(lower[0] - upper[0]) + max(upper[2], lower[2])
		h = lower[1] - upper[1] + lower[3]
		rects.append((x,y,w,h))
	return rects

def find_pairs(rects, offset):
	pairs = []
	changed = []
	# Fix contours for side by side
	for i in range(len(rects)):
		for j in range(len(rects)):
			if j <= i:
				continue
			c_1 = rects[i]
			c_2 = rects[j]
			if abs(c_1[0]-c_2[0]) <= offset:
				pairs.append([c_1,c_2])
				changed.append(c_1)
				changed.append(c_2)
	return pairs, changed

def rec_from_image(fileLocation, rawdata, hogdata):
	image = cv2.imread(fileLocation,0)
	# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	white = np.copy(image)
	for i in range(len(white)):
		for j in range(len(white[0])):
			white[i][j] = 255

	blur = cv2.GaussianBlur(image, (3, 3), 0)
	frameDelta = cv2.absdiff(white, blur)
	thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]
	thresh = cv2.dilate(thresh, None, iterations=2)
	_, cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)

	rects = []
	for c in cnts:
		rects.append(cv2.boundingRect(c))

	# Fix contours for up and down
	pairs, changed = find_pairs(rects, 10)
	rects = pair_pairs(pairs, rects)

	for c in changed:
		if c in rects:
			rects.remove(c)

	pairs, changed = find_pairs(rects, 50)
	rects = pair_pairs(pairs, rects)

	for c in changed:
		if c in rects:
			rects.remove(c)

	knnRawImage = np.copy(image)
	knnHOGImage = np.copy(image)
	SVMRawImage = np.copy(image)
	SVMHOGImage = np.copy(image)


	train_labels = np.repeat(lettersN, 10)[:,np.newaxis]
	trainRaw = make_usable_data(rawdata, 10, 0)
	trainHOG = make_usable_data(hogdata, 10, 0)

	# ### Train kNN-raw
	knnRaw = cv2.ml.KNearest_create()
	knnRaw.train(trainRaw, cv2.ml.ROW_SAMPLE, train_labels)
	print "kNN-Raw trained"
	# ### Train kNN-HOG
	knnHOG = cv2.ml.KNearest_create()
	knnHOG.train(trainHOG, cv2.ml.ROW_SAMPLE, train_labels)
	print "kNN-HOG trained"
	# ### Train SVM-raw
	svmRAW = cv2.ml.SVM_create()
	svmRAW.setGamma(SVM_GAMMA)
	svmRAW.setC(SVM_C)
	svmRAW.setKernel(cv2.ml.SVM_LINEAR)
	svmRAW.setType(cv2.ml.SVM_C_SVC)
	ok = svmRAW.train(trainRaw,cv2.ml.ROW_SAMPLE,train_labels)
	print "SVM-HOG trained"
	# ### Train SVM-raw
	svmHOG = cv2.ml.SVM_create()
	svmHOG.setGamma(SVM_GAMMA)
	svmHOG.setC(SVM_C)
	svmHOG.setKernel(cv2.ml.SVM_LINEAR)
	svmHOG.setType(cv2.ml.SVM_C_SVC)
	ok = svmHOG.train(trainHOG,cv2.ml.ROW_SAMPLE,train_labels)
	print "SVM-HOG trained"

	for rect in rects:
		# compute the bounding box for the contour, draw it on the frame,
		# and update the text
		(x, y, w, h) = rect
		rectImage = image[y:h+y,x:w+x]
		rectImage = cv2.resize(rectImage, (50,50))
		thresh = cv2.threshold(rectImage,100,255,cv2.THRESH_BINARY_INV)[1]
		thresh = cv2.dilate(thresh, np.ones((3,3),np.uint8), iterations=2)
		cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

		test_raw = np.array([thresh])
		deskewed = [map(deskew,row) for row in test_raw]
		test_hogdata = [map(hog,row) for row in deskewed]
		test_hogdata = np.array(test_hogdata)

		test_raw = test_raw.reshape(-1,2500).astype(np.float32)
		test_hogdata = test_hogdata.reshape(-1,3200).astype(np.float32)

		ret,result,neighbours,dist = knnRaw.findNearest(test_raw, k=4)
		cv2.putText(knnRawImage,letters[int(result[0][0])],(x+w/2,y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,255,0),2)

		ret,result,neighbours,dist = knnHOG.findNearest(test_hogdata, k=4)
		cv2.putText(knnHOGImage,letters[int(result[0][0])],(x+w/2,y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,255,0),2)

		result = svmRAW.predict(test_raw)
		cv2.putText(SVMRawImage,letters[int(result[1][0][0])],(x+w/2,y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,255,0),2)

		result = svmHOG.predict(test_hogdata)
		cv2.putText(SVMHOGImage,letters[int(result[1][0][0])],(x+w/2,y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,255,0),2)


	cv2.imshow("image", image)
	cv2.imshow("knnRawImage", knnRawImage)
	cv2.imshow("knnHOGImage", knnHOGImage)
	cv2.imshow("SVMRawImage", SVMRawImage)
	cv2.imshow("SVMHOGImage", SVMHOGImage)


x, hogdata = generate_image_data()
# test_kNN_accuracy(x,8,2, "kNN: Raw-Data accuracy: ") # kNN with raw pixel data
# test_kNN_accuracy(hogdata,8,2, "kNN: HOG data accuracy: ") # kNN with HOG data
# test_SVM_accuracy(x,8,2, "SVM: Raw-Data data accuracy: ") # SVM with raw pixel data
# test_SVM_accuracy(hogdata,8,2, "SVM: HOG data accuracy: ") # SVM with HOG data

testFile = "../data/long/nihon.png"
rec_from_image(testFile, x, hogdata)

cv2.waitKey(0)
