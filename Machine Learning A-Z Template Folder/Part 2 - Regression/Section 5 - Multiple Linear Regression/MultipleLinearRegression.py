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




