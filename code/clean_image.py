# import the necessary packages
import argparse
import datetime
import imutils
import time
import cv2
import numpy as np
from matplotlib import pyplot as plt
from os import listdir
from os.path import isfile, join

folder = "../data/templates/jis/"
files = [f for f in listdir(folder) if isfile(join(folder, f))]
whiteRemovedTop = []
whiteRemovedBottom = []
whiteRemovedLeft = []
whiteRemovedRight = []
countFiles = len(files)
i = 0.0
for file in files:
	i = i+1
	image = cv2.imread(folder + file, 0)
	image = image[2:len(image)-1]
	blackRemoved = []
	for lineNr in range(len(image)):
		blackRemoved.append(image[lineNr][2:len(image[lineNr])-1])

	### Remove white rows on the top
	for lineNr in range(len(blackRemoved)):
		badPixels = 0
		for px in blackRemoved[lineNr]:
			if px < 210:
				badPixels = badPixels+1

		if badPixels != 0 and badPixels < len(blackRemoved[lineNr]) / 2:
			lastOkRow = lineNr-1
			if lastOkRow > 0:
				whiteRemovedTop = blackRemoved[lastOkRow:]
				break
			else:
				whiteRemovedTop = blackRemoved
				break

	### Remove white rows on the bottom
	for lineNr in reversed(range(len(whiteRemovedTop))):
		badPixels = 0
		for px in whiteRemovedTop[lineNr]:
			if px < 210:
				badPixels = badPixels+1

		if badPixels != 0 and badPixels < len(whiteRemovedTop[lineNr]) / 2:
			lastOkRow = lineNr+1
			if lastOkRow < len(whiteRemovedTop):
				whiteRemovedBottom = whiteRemovedTop[:lastOkRow]
				break
			else:
				whiteRemovedBottom = whiteRemovedTop
				break

	### Remove white column on the left
	for pixelNr in range(len(whiteRemovedBottom[0])):
		badPixels = 0
		for lineNr in range(len(whiteRemovedBottom)):
			if whiteRemovedBottom[lineNr][pixelNr] < 210:
				badPixels = badPixels +1

		if badPixels != 0 and badPixels < len(whiteRemovedBottom) / 2:
			lastOkColumn = pixelNr-1
			if lastOkColumn > 0:
				for line in whiteRemovedBottom:
					whiteRemovedLeft.append(line[lastOkColumn:])
				break
			else:
				whiteRemovedLeft = whiteRemovedBottom
				break

	### Remove white column on the right
	for pixelNr in reversed(range(len(whiteRemovedLeft[0]))):
		badPixels = 0
		for lineNr in range(len(whiteRemovedLeft)):
			if whiteRemovedLeft[lineNr][pixelNr] < 210:
				badPixels = badPixels +1

		if badPixels != 0 and badPixels < len(whiteRemovedLeft) / 2:
			lastOkColumn = pixelNr+1
			if lastOkColumn < len(whiteRemovedLeft[0]):
				for line in whiteRemovedLeft:
					whiteRemovedRight.append(line[:lastOkColumn])
				break
			else:
				whiteRemovedRight = whiteRemovedLeft
				break

	whiteRemovedRight = cv2.resize(np.array(whiteRemovedRight), (50,50))
	cv2.imwrite(folder+file, whiteRemovedRight)
	whiteRemovedTop = []
	whiteRemovedBottom = []
	whiteRemovedLeft = []
	whiteRemovedRight = []
	print (i/countFiles)*100.0
