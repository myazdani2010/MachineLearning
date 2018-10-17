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

dataset = pd.read_csv('data.csv')

# make the matrix of features
X = dataset.iloc[:,:-1].values

# make variable vector 
y = dataset.iloc[:, 3]





#-----------------------------------------------
##### Steps 2- Deal With Missing Dataand #####
##### and 3- Categorical Data not required #####
#-----------------------------------------------





#------------------------------------------
##### 4- Split to Train and Test  #####
#------------------------------------------

from sklearn.cross_validation import train_test_split

# feed train and test sets based on 80/20 %
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0) 





#------------------------------------------
##### 5- Feature Scaling #####
#------------------------------------------
# feature scalling is very important for linear regression because it workes based on EEuclidean Distance
"""
from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
"""




