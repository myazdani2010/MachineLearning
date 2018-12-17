# Logistic Regression
# Data set contains Social Network Users. 
# Features: User ID, Gender, Age, EstimatedSalary, Purchased
#
# Problem: A car company has recently launched a new expensive SUV. We want to predict 
# which one of these users in the social Media most likely will purchase the car.
#
# Solution: we are considering features: Age and EstimatedSalary.


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Importing the dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, 4].values


# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)


# Predicting the Test set result
y_pred = classifier.predict(X_test)