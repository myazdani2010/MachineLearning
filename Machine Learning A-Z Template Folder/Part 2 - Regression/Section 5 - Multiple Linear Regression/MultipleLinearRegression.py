"""

This template will be required for all the ML practices to make initial steps ready.
There are only few parts need to be updated while trainning a new model in any dataset.
Note that feature scalling is not required for every ML trainig, hence its commented.

"""

# import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#------------------------------------------
##### 1- Import Dataset #####
#------------------------------------------

dataset = pd.read_csv('50_Startups.csv')

# make the matrix of independent variable X
X = dataset.iloc[:,:-1].values

# make the vector of dependent variable y
y = dataset.iloc[:, 4]





#------------------------------------------
##### 3- Categorical Data  #####
#------------------------------------------

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelEncoder_X = LabelEncoder()

# encode the State column
X[:,3] = labelEncoder_X.fit_transform(X[:,3])

# for categorial variables we must use Dummy Encoding 
# make State column as categorical feature 
oneHotEncoder = OneHotEncoder(categorical_features=[3])
X = oneHotEncoder.fit_transform(X).toarray()

# now we should have the State column converted to 3 new numerical columns which contains 0s and 1s

# Avoid Dummy Variable trap by removing any of the 3 new created columns from the Categorial column State
 X = X[:, 1:]

