import keras
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras.models import load_model

from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split

import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error,  r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
import numpy as np
from statistics import median, mean
from math import sqrt

#weights file
filename = 'regression_model.h5'

dataset = "OnlineNewsPopularityRegression.csv"

#better choice than np.loadtxt
df = pd.read_csv(dataset)
#first 2 columns are meta data, not used for training
df = df.iloc[:, 2:]

#Scaling/standardizing data
scaler = MinMaxScaler()

X=df.drop('shares', axis=1)
y=df['shares']
svd = TruncatedSVD(n_components=10)
arr = svd.fit_transform(X)
arr = pd.DataFrame(arr)
new_X = pd.concat([X, arr], axis=1)
new_X[new_X.columns] = scaler.fit_transform(new_X[new_X.columns])

new_X = new_X.as_matrix()
y = y.as_matrix()

train_X, test_X, train_y, test_y = train_test_split(new_X, y, train_size=0.7, random_state=123)

# Save the number of columns in predictors: n_cols
n_cols = new_X.shape[1]

"""
# Set up the model: model
model = Sequential()

# Add the first layer
model.add(Dense(68, activation='relu', input_shape=(n_cols,)))
# Add the second layer
model.add(Dense(68, activation='relu'))
# Add the third layer
model.add(Dense(68, activation='relu'))

# Add the output layer
model.add(Dense(1))

#compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Verify that model contains information from compiling
#print("Loss function: " + model.loss)

EarlyStopper= EarlyStopping(monitor='loss', patience=3)

#model_fitting
model.fit(train_X, train_y, epochs=100, callbacks=[EarlyStopper])

# save the model to disk
model.save(filename)  # creates a HDF5 file 
"""

# load the model from disk
model = load_model(filename)

predictions = model.predict(test_X)

#metrics
print("Root mean squared error: %f" % sqrt(mean_squared_error(test_y, predictions))) #14494
print("Mean absolute error: %f" % mean_absolute_error(test_y, predictions)) #3074
