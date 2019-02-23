from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import data_extract
import models
import numpy as np
import os
import argparse
import locale

path = "/Users/tanmaygulati/Work/ML:DL/PyImageSearch/Keras_Series_House_Price_Prediction/Houses-dataset-master/Houses_Dataset/HousesInfo.txt"

print("Loading Dataset: ")

df = data_extract.load_house_attrib(path)

print("Training and Test Split \n")

train, test = train_test_split(df, test_size = 0.25, random_state = 42)

maxPrice = train["price"].max()
trainY = train["price"] / maxPrice
testY = test["price"] / maxPrice

print("Data Cleaning : ")

(trainX, testX) = data_extract.process_house_attributes(df, train, test)

print(trainX.shape)
model = models.create_mlp(trainX.shape[1], regress = True)
opt = Adam(lr = 1e-4, decay = 1e-3 / 200)
model.compile(loss = "mean_absolute_percentage_error", optimizer = opt)

print("\n\n\nTraining\n\n")

model.fit(trainX, trainY, validation_data = (testX, testY), epochs = 200, batch_size = 8)

print("Predicting House Prices")
preds = model.predict(testX)

diff = preds.flatten() - testY
percentDiff = (diff / testY) * 100
absPercentDiff = np.abs(percentDiff)

mean = np.mean(absPercentDiff)
std = np.std(absPercentDiff)

print(mean)

