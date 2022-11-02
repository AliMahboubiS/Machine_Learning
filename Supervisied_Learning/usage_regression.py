from tkinter.ttk import Style
from turtle import color
from sklearn.model_selection import train_test_split
from sklearn.linear_model import  LinearRegression
from sklearn.metrics  import r2_score, mean_squared_error
from  matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import os 

CURRENT_DIR = os.path.dirname(__file__)
file_path   = os.path.join(CURRENT_DIR, "../data/bike-share.csv")

bike_data =  pd.read_csv(file_path)
# feature scale
bike_data['day'] =  pd.DatetimeIndex(bike_data['dteday']).day

# print(bike_data.head(35))

numeric_features = ['temp', 'atemp', 'hum', 'windspeed']
bike_data[numeric_features + ['rentals']].describe()

# get the label
label = bike_data['rentals']


# create a figure for 2 subplots (2 rows, 1 column)
fig, ax = plt.subplots(2, 1, figsize=(9, 12))

# plot histogram
ax[0].hist(label, bins=100)
ax[0].set_ylabel('Frequency')

# add line for mean, median
ax[0].axvline(label.mean(), color='grey', linestyle='dashed', linewidth=2)
ax[0].axvline(label.median(), color='grey', linestyle='dashed', linewidth=2)

# plot boxplot
ax[1].boxplot(label, vert=False)
ax[1].set_xlabel('Rentals')

fig.suptitle('Rental Distribution')

fig.show()

# plot histogram for each column
for col in numeric_features:
    fig = plt.figure(figsize=(9, 6))
    ax = fig.gca()
    feature =  bike_data[col]
    feature.hist(bins=100, ax=ax)
    ax.axvline(feature.mean(), color='gray', linestyle='dashed', linewidth=2)
    ax.axvline(feature.median(), color='gray', linestyle='dashed', linewidth=2)
    ax.set_title(col)

# plt.show()

# plot categiorical features
categorical_features = ['season', 'mnth', 'holiday', 'weekday', 'workingday', 'weathersit']

for col in categorical_features:
    counts =  bike_data[col].value_counts().sort_index()
    fig = plt.figure(figsize=(9,6))
    ax =  fig.gca()
    counts.plot.bar(ax= ax,  color='steelblue')
    ax.set_title(col + 'counts')
    ax.set_xlabel(col)
    ax.set_ylabel('Frequency')

# plt.show()

#show the relation  between numeric features and rentals

for col in numeric_features:
    fig = plt.figure(figsize=(9, 6))
    ax = fig.gca()
    feature = bike_data[col]
    label = bike_data['rentals']
    correlation = feature.corr(label)
    plt.scatter(x=feature, y=label)
    plt.xlabel(col)
    plt.ylabel('Bike Rentals')
    ax.set_title('rentals vs ' + col + '- correlation: ' + str(correlation))
#plt.show()


# plot a boxplot for the label by each categorical feature
for col in categorical_features:
    fig = plt.figure(figsize=(9, 6))
    ax = fig.gca()
    bike_data.boxplot(column = 'rentals', by = col, ax = ax)
    ax.set_title('Label by ' + col)
    ax.set_ylabel("Bike Rentals")
# plt.show()


# go for training section
# first we split the data to X, y
# Separate features and labels
X, y = bike_data[['season','mnth', 'holiday','weekday','workingday','weathersit','temp', 'atemp', 'hum', 'windspeed']].values, bike_data['rentals'].values
print('Features:',X[:10], '\nLabels:', y[:10], sep='\n')

X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.30, random_state=0)
print('Training set: %d rows\nTest set: %d rows'%(X_train.shape[0], X_test.shape[0]))

# fit the linear regression to or model
model =  LinearRegression().fit(X_train, y_train)
print(model)

# check and evaluate the test file
prediction = model.predict(X_test)
np.set_printoptions(suppress=True)
print('Predictedt label : ', np.round(prediction)[:10])
print('Actual lable : ', y_test[:10])

# show y_real vs y_grouth in visualization

plt.scatter(y_test, prediction)
plt.xlabel('Actual Labels')
plt.ylabel('Predicted Labels')
plt.title('Daily Bike Share Predictions')
# overlay the regression line
z = np.polyfit(y_test, prediction, 1)
p = np.poly1d(z)
plt.plot(y_test,p(y_test), color='magenta')
plt.show()

# calculate the errors metrics
mse = mean_squared_error(y_test, prediction)
print("MSE:", mse)

rmse = np.sqrt(mse)
print("RMSE:", rmse)

r2 = r2_score(y_test, prediction)
print("R2:", r2)
test = 0