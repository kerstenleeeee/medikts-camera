import pickle
import cv2
import os
import re
import csv
import numpy as np
from skimage.feature import hog

# training models
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.neural_network import MLPClassifier

# module for arduino
import serial

def flip():
	imageList = []
	imageDir = "sample-train/"

	count = 0
	for filename in os.listdir(imageDir):
		imageList.append(os.path.join(imageDir, filename))
	for imagePath in imageList:
		count = count + 1
		img = cv2.imread(imagePath)
		newImg = np.fliplr(img)
		strImg = re.split("[/()]", imagePath)
		print(strImg)
		outname = "sample-train/{}({} flip).jpg".format(strImg[1], str(count))
		cv2.imwrite(outname, newImg)

def grayscale():
	imageList = []
	imageDir = "sample-train/"

	count = 0
	for filename in os.listdir(imageDir):
		imageList.append(os.path.join(imageDir, filename))
	for imagePath in imageList:
		count = count + 1
		img = cv2.imread(imagePath)
		newImg = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
		strImg = re.split("[/()]", imagePath)
		print(strImg)
		outname = "sample-train/{}({} gray).jpg".format(strImg[1], str(count))
		cv2.imwrite(outname, newImg)

def meansub():
	imageList = []
	imageDir = "sample-train/"

	count = 0
	for filename in os.listdir(imageDir):
		imageList.append(os.path.join(imageDir, filename))
	for imagePath in imageList:
		count = count + 1
		img = cv2.imread(imagePath)
		mean = np.mean(img)
		newImg = img - mean
		strImg = re.split("[/()]", imagePath)
		print(strImg)
		outname = "sample-train/{}({} gray).jpg".format(strImg[1], str(count))
		cv2.imwrite(outname, newImg)

def resize():
	if not "sample-train-final" in os.listdir("."):
		os.mkdir("sample-train-final")

	imageList = []
	imageDir = "sample-train/"

	count = 0
	for filename in os.listdir(imageDir):
		imageList.append(os.path.join(imageDir, filename))
	for imagePath in imageList:
		count = count + 1
		img = cv2.imread(imagePath)
		newImg = cv2.resize(img, (128,128), interpolation = cv2.INTER_AREA)
		strImg = re.split("[/()]", imagePath)
		print(strImg)
		outname = "sample-train-final/{}({}).jpg".format(strImg[1], str(count))
		cv2.imwrite(outname, newImg)

	print("Done.")

def preprocess():
	while(1):
		print("Choose")
		print("(1) Flip")
		print("(2) Grayscale")
		print("(3) Mean Subtraction")
		print("(4) Resize")
		
		choice = int(input())

		if choice == 1:
			print("\nFlip...")
			flip()
		elif choice == 2:
			print("\nColoring...")
			grayscale()
		elif choice == 3:
			print("\nSubracting...")
			meansub()
		elif choice == 4:
			print("\nResizing...")
			resize()
		else:
			break

def featureVector():
	hgd = cv2.HOGDescriptor()

	imageList = []
	imageDir = "sample-train-final/"

	for filename in os.listdir(imageDir):
		imageList.append(os.path.join(imageDir, filename))

	fv = np.zeros((len(imageList), 34020))

	# print(fv.shape)

	for imageIndex, imagePath in enumerate(imageList):
		img = cv2.imread(imagePath)
		val = hgd.compute(img)
		#print(len(val))

		for i in range(34020):
			fv[imageIndex][i] = val[i]

	csvFile = open("datasetTrain.csv", "w", newline='')
	writer = csv.writer(csvFile)
	for i in range(len(imageList)):
		writer.writerow(fv[i])

	print("Done.")

def train():
	labelsMatrix = np.zeros(220)
	with open("labels.txt", "r") as labels:
		for index, line in enumerate(labels):
			line = line.split()
			labelsMatrix[index] = int(line[0])

	datasetTrain = np.genfromtxt("datasetTrain.csv", delimiter = ",")
	model = LinearSVC()
	# model = MLPClassifier()
	# model = LinearSVC()
	model.fit(datasetTrain, labelsMatrix)
	filename = "trainModel.sav"
	pickle.dump(model, open(filename, "wb"))
	print("Done.")

def test():
	loadModel = pickle.load(open("trainModel.sav", "rb"))

	img = cv2.imread("sample-test/test2.jpg")
	newImg = cv2.resize(img, (128,128), interpolation = cv2.INTER_AREA)
	hgd = cv2.HOGDescriptor()
	features = []
	val = hgd.compute(newImg)
	features.append(val)
	features = np.asarray(features)
	features = features.reshape(len(features), -1)
	#print(features.shape)

	pred = loadModel.predict(features)

	############ checker #################
	if int(pred) == 1:
		print("\nFenoflex")
	elif int(pred) == 2:
		print("\nZylet")
	elif int(pred) == 3:
		print("\nRosuvaz")
	elif int(pred) == 4:
		print("\nLung Caire Plus")
	elif int(pred) == 5:
		print("\nGlycoair")

if __name__ == '__main__':
	print("Choose")
	print("(1) Preprocess")
	print("(2) Feature Vector")
	print("(3) Train Model")
	print("(4) Test")
	
	choice = int(input())

	if choice == 1:
		print("\nPreprocessing...")
		preprocess()
	elif choice == 2:
		print("\nCreating feature vector...")
		featureVector()
	elif choice == 3:
		print("\nTraining...")
		train()
	elif choice == 4:
		print("\nTesting...")
		test()
	else:
		print("\nERROR")