import keras
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.models import load_model

from sklearn.model_selection import train_test_split
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import cross_val_score

import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error,  r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
import numpy as np
from statistics import median, mean
from math import sqrt

#weights file
filename = 'classification_model.h5'

dataset = "OnlineNewsPopularityClassification.csv"

#better choice than np.loadtxt
df = pd.read_csv(dataset)
#first 2 columns are meta data, not used for training
df = df.iloc[:, 2:]

#Scaling/standardizing data
scaler = MinMaxScaler()
#scaler = MaxAbsScaler()

X=df.drop('shares', axis=1)
#X[X.columns] = scaler.fit_transform(X[X.columns])
y=to_categorical(df['shares'])

svd = TruncatedSVD(n_components=10)
arr = svd.fit_transform(X)
arr = pd.DataFrame(arr)
new_X = pd.concat([X, arr], axis=1)
new_X[new_X.columns] = scaler.fit_transform(new_X[new_X.columns])

new_X = new_X.as_matrix()

# Save the number of columns in predictors: n_cols
n_cols = new_X.shape[1]

# Set up the model: model
model = Sequential()

# Add the first layer
model.add(Dense(100, activation='relu', input_shape=(n_cols,)))
# Add the second layer
model.add(Dense(70, activation='relu'))
model.add(Dense(32, activation='relu'))

# Add the output layer
model.add(Dense(2, activation='softmax'))

EarlyStopper= EarlyStopping(monitor='loss', patience=2)

#compile the model
model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

# Verify that model contains information from compiling
#print("Loss function: " + model.loss)

#UnboundLocalError: local variable 'arrays' referenced before assignment
#Solution: Convert dataframe to numpy array

#model_fitting
model.fit(new_X, y, epochs=100, validation_split=0.3, callbacks=[EarlyStopper])

# save the model to disk
#model.save(filename)  # creates a HDF5 file 

# load the model from disk
#model = load_model(filename)
