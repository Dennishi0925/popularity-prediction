import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso, BayesianRidge
from sklearn.metrics import mean_squared_error, mean_absolute_error,  r2_score
from sklearn.ensemble import RandomForestRegressor, VotingClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from sklearn.decomposition import TruncatedSVD
import numpy as np
from statistics import median, mean
from math import sqrt
import pickle

#weights file
filename = 'finalized_model.sav'

dataset = "OnlineNewsPopularityRegression.csv"
#test_dataset ="C:/Users/Israr/Dropbox/Semester VII/DSc/Project Research/OnlineNewsPopularity/abc.csv" #worsens the accuracy

#better choice than np.loadtxt
df = pd.read_csv(dataset)
#first 2 columns are meta data, not used for training
df = df.iloc[:, 2:]

n_samples, n_features = df.shape

#Scaling/standardizing data
scaler = MinMaxScaler()

X=df.drop('shares', axis=1)
X=X.drop('n_non_stop_words', axis=1)
X[X.columns] = scaler.fit_transform(X[X.columns])
y=df['shares']

svd = TruncatedSVD(n_components=10)
arr = svd.fit_transform(X)
arr = pd.DataFrame(arr)
new_X = pd.concat([X, arr], axis=1)
new_X[new_X.columns] = scaler.fit_transform(new_X[new_X.columns])


#train_X, test_X, train_y, test_y = train_test_split(new_X, y, train_size=0.7, random_state=123)


train = df.iloc[:30000, :]
train_X = train.iloc[:, :58]
train_y = train.iloc[:, 58:]

test = df.iloc[30000:, :]
test_X = test.iloc[:, :58]

test_y = test.iloc[:, 58:]

predictions = np.zeros((test_y.shape))
#model_fitting
#All models
bay_rid = BayesianRidge()
bay_rid.fit(train_X, train_y)
bay_predictions = bay_rid.predict(test_X)
bay_predictions = bay_predictions.reshape(9644,1)

predictions += bay_predictions
lasso = Lasso()
lasso.fit(train_X, train_y)
lasso_predictions = lasso.predict(test_X)
lasso_predictions = lasso_predictions.reshape(9644,1)

predictions += lasso_predictions

ridge = Ridge(alpha=0.5)
ridge.fit(train_X, train_y)
predictions += ridge.predict(test_X)

lin_reg = LinearRegression()
lin_reg.fit(train_X, train_y)
predictions += lin_reg.predict(test_X)


#model = RandomForestRegressor(n_estimators=50) #promising
#print(model.feature_importances_)
#threshold = 1/len(model.feature_importances_)

# save the model to disk
#pickle.dump(model, open(filename, 'wb'))

"""

# load the model from disk
model = pickle.load(open(filename, 'rb'))
result = model.score(test_X, test_y)
print(result)

"""

#predictions = model.predict(test_X)
#print(max(predictions))
"""
keys = new_X.keys()
count = 0
for i in range(len(model.feature_importances_)):
    val = model.feature_importances_[i]
    if val > threshold:
        count+=1
        try:
            print(keys[i])
        except:
            print("Error")
        print(str(val))
        #df = df.drop(keys[i], axis=1)

print(count)
"""
#metrics
predictions /= 4
#print(predictions)
print("Root mean squared error: %f" % sqrt(mean_squared_error(test_y, predictions)))
print("Mean absolute error: %f" % mean_absolute_error(test_y, predictions))
#predictions.to_csv("my_predictions.csv")

errors = []
for i in range(0, len(predictions)):
    errors.append(abs(test_y.iloc[i] - predictions[i]))
    #print("Predicted: " + str(predictions[i])+"  Actual: "+str(test_y.iloc[i]))

#print(mean(list(errors)))
"""
print("Before - error median: %.2f" % median(errors))
errors = sorted(errors)
errors = errors[100:-100]
print(mean(errors))
"""


"""
#dropping less important columns
keys = df.keys()
count = 0
for i in range(len(model.feature_importances_)):
    val = model.feature_importances_[i]
    if val <= 0.001:
        count+=1
        #print(keys[i]+" "+str(val))
        df = df.drop(keys[i], axis=1)

#print(df.shape)
#1/60 = 0.0167 assuming equally likely variables
#feature importance of n_non_stop_words == 0.0 which is weird since it seems to be an imp feature

X=df.drop(' shares', axis=1)
X[X.columns] = scaler.fit_transform(X[X.columns])
#print(X.iloc[0])
#print(df.shape)
y=df[' shares']


train_X, test_X, train_y, test_y = train_test_split(X, y, train_size=0.7, random_state=123)

#model_fitting
model = RandomForestRegressor(max_depth=10, random_state=0) #promising
model.fit(train_X, train_y)


predictions = model.predict(test_X)

#metrics
print("Mean squared error: %f" % mean_squared_error(test_y, predictions))
print("Mean absolute error: %f" % mean_absolute_error(test_y, predictions))

errors = []
for i in range(0, len(predictions)):
    errors.append(abs(test_y.iloc[i] - predictions[i]))
    #print("Predicted: " + str(predictions[i])+"  Actual: "+str(test_y.iloc[i]))

print(errors)
print("After - error median: %.2f" % median(errors))

median_data = median(errors)
print(sorted(errors))
"""
