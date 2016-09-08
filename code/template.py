# import the necessary packages
import argparse
import datetime
import imutils
import time
import cv2
import numpy as np
from matplotlib import pyplot as plt


image = cv2.imread("../data/templates/wo-zo/file-page14-crop.jpg", 0)

# cv2.rectangle(image, (10,5), (130,125), (0, 255, 0), 2)
a_ni = ["a", "i", "u","e","o","ka","ki","ku","ke","ko","sa","shi","su","se","so","ta","chi","tsu","te","to", "na", "ni"]
nu_wa = ["nu", "ne", "no", "ha", "hi", "fu", "he", "ho", "ma", "mi", "mu", "me", "mo", "ya", "yu", "yo", "ra", "ri", "ru", "re", "ro", "wa"]
wo_zo = ["wo", "n", "ba", "bi", "bu", "be", "bo", "pa", "pi", "pu", "pe", "po", "ga", "gi", "gu", "ge", "go", "za", "ji(shi)", "zu", "ze", "zo"]
da_do = ["da", "ji", "du", "de", "do"]
print len(wo_zo)
x=10
y=8
w=130
h=124
offsetX = 4
offsetY = 20
nameInit = 176
name = nameInit
# cv2.rectangle(image, (x,y), (w,h), (0, 255, 0), 2)
for j in range(22):
	if j == 18:
		for i in range(14):
			if i != 0:
				if i > 8:
					tmp_x = x+i*w+offsetX*i+4
					tmp_w = i*w+w+offsetX*i+4
					cv2.rectangle(image, (tmp_x,y), (tmp_w,h), (0, 255, 0), 2)
					cv2.imwrite("../data/templates/jis/"+wo_zo[j]+"_"+str(name)+".png", image[y:h,tmp_x:tmp_w])
					# print str(tmp_x) + " -- " + str(tmp_w)+ " __________ " + str(y) + " -- " + str(h)
				else:
					tmp_x = x+i*w+offsetX*i
					tmp_w = i*w+w+offsetX*i
					cv2.rectangle(image, (tmp_x,y), (tmp_w,h), (0, 255, 0), 2)
					cv2.imwrite("../data/templates/jis/"+wo_zo[j]+"_"+str(name)+".png", image[y:h,tmp_x:tmp_w])
					# print str(tmp_x) + " -- " + str(tmp_w) + " __________ " + str(y) + " -- " + str(h)
			name = name+1
	name = nameInit
	y=y+130
	h=h+130
	if j == 6:
		y=y+5
		h=h+5
	if j == 10:
		y=y+5
		h=h+5
	if j == 14:
		y=y+6
		h=h+6
	if j == 18:
		y=y+5
		h=h+5

# cv2.imshow("real", image)
cv2.imwrite("test.png", image)

cv2.waitKey(0)