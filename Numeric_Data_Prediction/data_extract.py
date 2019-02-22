from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import glob
import cv2
import os

def load_house_attrib(path):
	cols = ["bedrooms", "bathrooms", "area", "zipcode", "price"]
	df = pd.read_csv(path, sep = " ", header = None, names = cols)
	print(df.head(5))

	zipcodes = df["zipcode"].value_counts().keys().tolist()
	counts = df["zipcode"].value_counts().keys()

	for (zipcode, count) in zip(zipcodes, counts):
		# cleaning the data for zipcodes having count of houses less than 25
		if count < 25:
			id = df[df["zipcode"] == zipcode].index
			df.drop(id, inplace = True)

		return df

def process_house_attributes(df, train, test):
	continous = ["bedrooms", "bathrooms", "area"]
	cs = MinMaxScaler()
	trainContinous = cs.fit_transform(train[continous])
	testContinous = cs.transform(test[continous])

	# one hot encoding the zipcode categorical data
	zipBinarizer = LabelBinarizer().fit(df["zipcode"])
	print(zipBinarizer)
	trainCategorical = zipBinarizer.transform(train["zipcode"])
	testCategorical = zipBinarizer.transform(test["zipcode"])

	trainX = np.hstack([trainCategorical, trainContinous])
	testX = np.hstack([testCategorical, testContinous])

	print(trainX[0], '\n', testX[0])

	return trainX, testX



# path = "/Users/tanmaygulati/Work/ML:DL/PyImageSearch/Keras_Series_House_Price_Prediction/Houses-dataset-master/Houses_Dataset/HousesInfo.txt"

# df = load_house_attrib(path)

# from sklearn.model_selection import train_test_split
# train, test = train_test_split(df, test_size = 0.25, random_state = 42)

# process_house_attributes(df, train, test)

