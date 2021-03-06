# import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#------------------------------------------
##### 1- Import Dataset #####
#------------------------------------------

dataset = pd.read_csv('data.csv')

# make the matrix of features
X = dataset.iloc[:,:-1].values

# make variable vector 
y = dataset.iloc[:, 3]





#------------------------------------------
##### 2- Deal With Missing Data #####
#------------------------------------------

from sklearn.preprocessing import Imputer

imputer = Imputer(missing_values="NaN", strategy="mean", axis=0)

# take columns 1 and 2 
imputer.fit(X[:, 1:3]) 

# replace the missing data by mean of that column
X[:, 1:3] = imputer.transform(X[:, 1:3])





#------------------------------------------
##### 3- Categorical Data  #####
#------------------------------------------

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelEncoder_X = LabelEncoder()

# convert the Country column to encoded numerical value 
X[:,0] = labelEncoder_X.fit_transform(X[:,0])

# for categorial variables we must use Dummy Encoding 
# make first column as categorical feature 
oneHotEncoder = OneHotEncoder(categorical_features=[0])
X = oneHotEncoder.fit_transform(X).toarray()

# Since y is boolean then we only need to convert to numerical data and no need for Dummy Encoding
labelEncoder_y = LabelEncoder() 
y = labelEncoder_y.fit_transform(y)





#------------------------------------------
##### 4- Split to Train and Test  #####
#------------------------------------------

from sklearn.cross_validation import train_test_split

# feed train and test sets based on 80/20 %
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0) 





#------------------------------------------
##### 5- Feature Scaling  #####
#------------------------------------------
# feature scalling is very important for linear regression because it workes based on EEuclidean Distance

from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)





