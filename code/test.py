# import the necessary packages
import argparse
import datetime
import imutils
import time
import cv2
import numpy as np
from random import shuffle
from matplotlib import pyplot as plt
from os import listdir
from os.path import isfile, join

letters = ["a","i","u","e","o","ka","ki","ku","ke","ko","sa","shi","su","se","so","fu","ha","hi","ho","he","ma","mi","mu","me","mo","n","na","ni","no","nu","ne","ra","ri","ru","re","ro","ta","chi","to","te","tsu","wa","wo","ya","yo","yu"]
lettersN = range(46)
filePrefixes = ["a","i","u","e","o","ka","ki","ku","ke","ko","sa","shi","su","se","so","fu","ha","hi","ho","he","ma","mi","mu","me","mo","n_","na","ni","no","nu","ne","ra","ri","ru","re","ro","ta","chi","to","te","tsu","wa","wo","ya","yo","yu", "da", "ji_", "du", "de", "do","zo","ji(shi)","zu","ze","zo","ba","bi","bu","be","bo","pa","pi","pu","pe","po", "ga","gi","gu","ge","go"]

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

	# Make images to np array
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
	print len(train)
	print len(train[0])

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

# TEST
def test_kNN_HOG_accuracy_full(test_amount):
	knnRaw = cv2.ml.KNearest_create()

	folder = "../data/templates/singles_50x50/"
	files = [f for f in listdir(folder) if isfile(join(folder, f))]
	fileCounts = []
	train = []
	train_labels = []
	test = []
	test_labels = []
	i = 1
	j = 0
	for name in filePrefixes:
		nameFiles = [k for k in files if k.startswith(name)]
		shuffle(nameFiles)
		fileCounts.append(len(nameFiles))
		for fileName in nameFiles:
			image = cv2.imread(folder + fileName,0)
			thresh = cv2.threshold(image,220,255,cv2.THRESH_BINARY_INV)[1]
			thresh = cv2.dilate(thresh, np.ones((3,3),np.uint8), iterations=2)
			thresh = np.array([thresh])
			deskewed = [map(deskew,row) for row in thresh]
			hogData = [map(hog,row) for row in deskewed]
			if(len(nameFiles) - test_amount < j):
				test.append(hogData)
				test_labels.append(filePrefixes.index(name))
			else:
				train.append(hogData)
				train_labels.append(filePrefixes.index(name))
			j = j+1
		j=0
		# print i/71.0 * 100.0
		i = i+1

	# print fileCounts
	# # Make images to np array
	x = np.array(train)
	x = x.reshape(-1,3200).astype(np.float32)
	knnRaw.train(x, cv2.ml.ROW_SAMPLE, np.array(train_labels))

	y = np.array(test)
	y = y.reshape(-1,3200).astype(np.float32)
	ret,result,neighbours,dist = knnRaw.findNearest(y,k=4)

	correct = 0
	for i in range(len(neighbours)):
		# print str(neighbours[i]) + " - " + str(test_labels[i]) + " - " + str(result[i])
		if test_labels[i] == result[i][0]:
			correct = correct + 1 

	accuracy = correct*100.0/result.size
	print "kNN - HOG: " + str(accuracy) + "%"

def test_kNN_RAW_accuracy_full(test_amount):
	knnRaw = cv2.ml.KNearest_create()

	folder = "../data/templates/singles_50x50/"
	files = [f for f in listdir(folder) if isfile(join(folder, f))]
	fileCounts = []
	train = []
	train_labels = []
	test = []
	test_labels = []
	i = 1
	j = 0
	for name in filePrefixes:
		nameFiles = [k for k in files if k.startswith(name)]
		shuffle(nameFiles)
		fileCounts.append(len(nameFiles))
		for fileName in nameFiles:
			image = cv2.imread(folder + fileName,0)
			thresh = cv2.threshold(image,220,255,cv2.THRESH_BINARY_INV)[1]
			thresh = cv2.dilate(thresh, np.ones((3,3),np.uint8), iterations=2)
			if(len(nameFiles) - test_amount <= j):
				test.append(thresh)
				test_labels.append(filePrefixes.index(name))
			else:
				train.append(thresh)
				train_labels.append(filePrefixes.index(name))
			j = j+1
		j=0
		# print i/71.0 * 100.0
		i = i+1

	# print fileCounts
	# # Make images to np array
	x = np.array(train)
	x = x.reshape(-1,2500).astype(np.float32)
	knnRaw.train(x, cv2.ml.ROW_SAMPLE, np.array(train_labels))

	y = np.array(test)
	y = y.reshape(-1,2500).astype(np.float32)
	ret,result,neighbours,dist = knnRaw.findNearest(y,k=4)
	correct = 0
	for i in range(len(neighbours)):
		# print str(neighbours[i]) + " - " + str(test_labels[i]) + " - " + str(result[i])
		if test_labels[i] == result[i][0]:
			correct = correct + 1 

	accuracy = correct*100.0/result.size
	print "kNN - RAW: " + str(accuracy) + "%"

def test_SVM_RAW_accuracy_full(test_amount):
	svm = cv2.ml.SVM_create()
	svm.setGamma(SVM_GAMMA)
	svm.setC(SVM_C)
	svm.setKernel(cv2.ml.SVM_LINEAR)
	svm.setType(cv2.ml.SVM_C_SVC)

	folder = "../data/templates/singles_50x50/"
	files = [f for f in listdir(folder) if isfile(join(folder, f))]
	fileCounts = []
	train = []
	train_labels = []
	test = []
	test_labels = []
	i = 1
	j = 0
	for name in filePrefixes:
		nameFiles = [k for k in files if k.startswith(name)]
		shuffle(nameFiles)
		fileCounts.append(len(nameFiles))
		for fileName in nameFiles:
			image = cv2.imread(folder + fileName,0)
			thresh = cv2.threshold(image,220,255,cv2.THRESH_BINARY_INV)[1]
			thresh = cv2.dilate(thresh, np.ones((3,3),np.uint8), iterations=2)
			if(len(nameFiles) - test_amount <= j):
				test.append(thresh)
				test_labels.append(filePrefixes.index(name))
			else:
				train.append(thresh)
				train_labels.append(filePrefixes.index(name))
			j = j+1
		j=0
		# print i/71.0 * 100.0
		i = i+1

	# print fileCounts
	# # Make images to np array
	x = np.array(train)
	x = x.reshape(-1,2500).astype(np.float32)
	ok = svm.train(x,cv2.ml.ROW_SAMPLE,np.array(train_labels))

	y = np.array(test)
	y = y.reshape(-1,2500).astype(np.float32)
	result = svm.predict(y)
	correct = 0
	for i in range(len(result[1])):
		# print str(test_labels[i]) + " - " + str(result[1][i][0])
		if test_labels[i] == result[1][i][0]:
			correct = correct + 1 

	accuracy = correct*100.0/result[1].size
	print "SVM - RAW: " + str(accuracy) + "%"

def test_SVM_HOG_accuracy_full(test_amount):
	svm = cv2.ml.SVM_create()
	svm.setGamma(SVM_GAMMA)
	svm.setC(SVM_C)
	svm.setKernel(cv2.ml.SVM_LINEAR)
	svm.setType(cv2.ml.SVM_C_SVC)

	folder = "../data/templates/singles_50x50/"
	files = [f for f in listdir(folder) if isfile(join(folder, f))]
	fileCounts = []
	train = []
	train_labels = []
	test = []
	test_labels = []
	i = 1
	j = 0
	for name in filePrefixes:
		nameFiles = [k for k in files if k.startswith(name)]
		shuffle(nameFiles)
		fileCounts.append(len(nameFiles))
		for fileName in nameFiles:
			image = cv2.imread(folder + fileName,0)
			thresh = cv2.threshold(image,220,255,cv2.THRESH_BINARY_INV)[1]
			thresh = cv2.dilate(thresh, np.ones((3,3),np.uint8), iterations=2)
			thresh = np.array([thresh])
			deskewed = [map(deskew,row) for row in thresh]
			hogData = [map(hog,row) for row in deskewed]
			if(len(nameFiles) - test_amount < j):
				test.append(hogData)
				test_labels.append(filePrefixes.index(name))
			else:
				train.append(hogData)
				train_labels.append(filePrefixes.index(name))
			j = j+1
		j=0
		# print i/71.0 * 100.0
		i = i+1

	# print fileCounts
	# # Make images to np array
	x = np.array(train)
	x = x.reshape(-1,3200).astype(np.float32)
	ok = svm.train(x,cv2.ml.ROW_SAMPLE,np.array(train_labels))

	y = np.array(test)
	y = y.reshape(-1,3200).astype(np.float32)
	result = svm.predict(y)
	correct = 0
	for i in range(len(result[1])):
		# print str(test_labels[i]) + " - " + str(result[1][i][0])
		if test_labels[i] == result[1][i][0]:
			correct = correct + 1 

	accuracy = correct*100.0/result[1].size
	print "SVM - HOG: " + str(accuracy) + "%"
####################################################################
# From https://gist.github.com/moshekaplan/5106221#file-test_surf-py
def filter_matches(kp1, kp2, matches, ratio = 0.75):
	mkp1, mkp2 = [], []
	for m in matches:
		if len(m) == 2 and m[0].distance < m[1].distance * ratio:
			m = m[0]
			mkp1.append( kp1[m.queryIdx] )
			mkp2.append( kp2[m.trainIdx] )
	kp_pairs = zip(mkp1, mkp2)
	return kp_pairs

def explore_match(win, img1, img2, kp_pairs, status = None, H = None):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    vis = np.zeros((max(h1, h2), w1+w2), np.uint8)
    vis[:h1, :w1] = img1
    vis[:h2, w1:w1+w2] = img2
    vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

    if H is not None:
        corners = np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]])
        corners = np.int32( cv2.perspectiveTransform(corners.reshape(1, -1, 2), H).reshape(-1, 2) + (w1, 0) )
        cv2.polylines(vis, [corners], True, (255, 255, 255))

    if status is None:
        status = np.ones(len(kp_pairs), np.bool_)
    p1 = np.int32([kpp[0].pt for kpp in kp_pairs])
    p2 = np.int32([kpp[1].pt for kpp in kp_pairs]) + (w1, 0)

    green = (0, 255, 0)
    red = (0, 0, 255)
    white = (255, 255, 255)
    kp_color = (51, 103, 236)
    for (x1, y1), (x2, y2), inlier in zip(p1, p2, status):
        if inlier:
            col = green
            cv2.circle(vis, (x1, y1), 2, col, -1)
            cv2.circle(vis, (x2, y2), 2, col, -1)
        else:
            col = red
            r = 2
            thickness = 3
            cv2.line(vis, (x1-r, y1-r), (x1+r, y1+r), col, thickness)
            cv2.line(vis, (x1-r, y1+r), (x1+r, y1-r), col, thickness)
            cv2.line(vis, (x2-r, y2-r), (x2+r, y2+r), col, thickness)
            cv2.line(vis, (x2-r, y2+r), (x2+r, y2-r), col, thickness)
    vis0 = vis.copy()
    for (x1, y1), (x2, y2), inlier in zip(p1, p2, status):
        if inlier:
            cv2.line(vis, (x1, y1), (x2, y2), green)

    cv2.imshow(win, vis)


  
def draw_matches(window_name, kp_pairs, img1, img2):
    """Draws the matches for """
    mkp1, mkp2 = zip(*kp_pairs)
    
    p1 = np.float32([kp.pt for kp in mkp1])
    p2 = np.float32([kp.pt for kp in mkp2])
    
    if len(kp_pairs) >= 4:
        H, status = cv2.findHomography(p1, p2, cv2.RANSAC, 5.0)
        #print '%d / %d  inliers/matched' % (np.sum(status), len(status))
    else:
        H, status = None, None
        #print '%d matches found, not enough for homography estimation' % len(p1)
    
    if len(p1):
		explore_match(window_name, img1, img2, kp_pairs, status, H)
####################################################################


# x, hogdata = generate_image_data()

# test_kNN_accuracy(x,8,2, "kNN: Raw-Data accuracy: ") # kNN with raw pixel data
# test_kNN_accuracy(hogdata,8,2, "kNN: HOG data accuracy: ") # kNN with HOG data
# test_SVM_accuracy(x,8,2, "SVM: Raw-Data data accuracy: ") # SVM with raw pixel data
# test_SVM_accuracy(hogdata,8,2, "SVM: HOG data accuracy: ") # SVM with HOG data

# testFile = "../data/long/nihon.png"
# rec_from_image(testFile, x, hogdata)

# Test with whole dataset.
# test_kNN_HOG_accuracy_full(80)
# test_kNN_RAW_accuracy_full(80)
# test_SVM_RAW_accuracy_full(80)
# test_SVM_HOG_accuracy_full(80)

folder = "../data/templates/singles_50x50/"
surf = cv2.SURF(100)
surf.extended = True
files = ["ba_86.png","go_172.png","po_64.png","hi_157.png","de_28.png","go_111.png","ho_91.png","ya_134.png","to_169.png","ki_166.png"]
matcher = cv2.BFMatcher(cv2.NORM_L2)


# for file in files:
# 	image = cv2.imread(folder + file, 0)
# 	kp, des = surf.detectAndCompute(image, None)
# 	print len(kp)
# 	img2 = cv2.drawKeypoints(image,kp,None,(255,0,0),4)

image1 = cv2.imread(folder + files[1], 0)
kp1, des1 = surf.detectAndCompute(image1, None)
# image1keypoints = cv2.drawKeypoints(image1,kp1,None,(255,0,0),4)

image2 = cv2.imread(folder + files[3], 0)
kp2, des2 = surf.detectAndCompute(image2, None)
# image2keypoints = cv2.drawKeypoints(image2,kp2,None,(255,0,0),4)

print len(kp1)
print len(kp2)
print len(des1)
print len(des2)

raw_matches = matcher.knnMatch(des1, trainDescriptors = des2, k = 2) #2
kp_pairs = filter_matches(kp1, kp2, raw_matches)
draw_matches("test", kp_pairs, image1, image2)

cv2.waitKey(0)
