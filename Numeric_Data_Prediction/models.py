# Multilayer Perceptron

from keras.models import Sequential
from keras.models.normalization import BatchNormalization
from keras.models.convolutional import Conv2D
from keras.models.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras.layers import Flatten
from keras.layers import Input
from keras.layers import Model

def create_mlp(dim, regress = False):
	model = Sequential()
	model.add(Dense(8, input_dim = dim, activation = "relu"))
	model.add(Dense(4, activation = "relu"))

	# if regression node should be added
	if regress == 1:
		model.add(Dense(1, activation = "linear"))

	return model


