from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import data_extract
import models
import numpy as np
import os
import argparse
import locale

path = "/Users/tanmaygulati/Work/ML:DL/PyImageSearch/Keras_Series_House_Price_Prediction/Houses-dataset-master/Houses_Dataset/HousesInfo.txt"
path2 = "/Users/tanmaygulati/Work/ML:DL/PyImageSearch/Keras_Series_House_Price_Prediction/Houses-dataset-master/Houses_Dataset/" # for images

print("Loading Dataset: ")
df = data_extract.load_house_attrib(path)

print("Loading Images : ")
images = data_extract.load_img(df, path2)
images = images / 255.0

print("Training and Test Split: ")
trainAttrX, testAttrX, trainImagesX, testImagesX = train_test_split(df, images, test_size=0.25, random_state=42)

maxPrice = trainAttrX["price"].max()
trainY = trainAttrX["price"] / maxPrice
testY = testAttrX["price"] / maxPrice

model = models.create_cnn(64, 64, 3, regress = True)
opt = Adam(lr = 1e-3, decay = 1e-3 / 200)
model.compile(loss="mean_absolute_percentage_error", optimizer = opt)

print("Training Model")
model.fit(trainImagesX, trainY, validation_data = (testImagesX, testY), epochs = 200, batch_size = 8)

print("Testing Model i.e. predicting house prices")
preds = model.predict(testImagesX)

diff = preds.flatten() - testY
percentDiff = (diff / testY) * 100
absPercentDiff = np.abs(percentDiff)

mean = np.mean(absPercentDiff)
std = np.std(absPercentDiff)

print("Predicting House Prices")
print("Mean House Price :\t", df["price"].mean())
print("Mean:\t", mean)
print("Standard Deviation:\t", std)



