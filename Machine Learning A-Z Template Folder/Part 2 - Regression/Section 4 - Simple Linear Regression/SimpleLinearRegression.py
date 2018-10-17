

"""
Dataset contains 30 records of employee's YearsExperience and Salary. Here we want to study the correlation between the 
years of experience and the salary.

"""


# import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#------------------------------------------
##### 1- Import Dataset #####
#------------------------------------------

dataset = pd.read_csv('Salary_Data.csv')

# create feature matix by excluding the last column
X = dataset.iloc[:,:-1].values

# create dependent variable vector from output column 
y = dataset.iloc[:, 1]





#------------------------------------------
##### 2- Split to Train and Test  #####
#------------------------------------------

from sklearn.cross_validation import train_test_split

# feed train and test sets based on 80/20 %
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0) 





#------------------------------------------
##### Feature Scaling is not required #####
#------------------------------------------
# the scaling will be applied by the python library
"""
from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
"""





#-----------------------------------------
##### Fitt simple linear regression #####  
##### to the training set ##### 
#-----------------------------------------

from sklearn.linear_model import LinearRegression

regressor = LinearRegression() # create machine 
regressor.fit(X_train, y_train) # train the machine with data





#------------------------------------------------------
##### Predicting the Test set result #####
#------------------------------------------------------
# here we use the test dataset to see how our trained machine is predecting the result
y_pred = regressor.predict(X_test) 














