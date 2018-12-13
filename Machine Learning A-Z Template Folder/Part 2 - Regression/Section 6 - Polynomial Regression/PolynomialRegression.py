# Polynomial Regression


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values


# Splitting the dataset into the Training set and Test set 
# this is not required because 
# 1- we dont have much data 
# 2- we want to use entire data to train the model


# Feature Scaling is applied by the library we are going to use 


# Fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)


# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=2) # this creates 2 extra feature 
X_poly = poly_reg.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)


# Visualising the Linear Regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg.predict(X), color = 'blue')
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('position level')
plt.ylabel('salary')
plt.show()

print("\nFrom the above graph we can see the blue line is the predection lines and red dots are actual values." + 
      "As we can see, the predection is not accurate since we can see some data far from the predection.\n\n" + 
      "We will use the Polynomial Regression to get the predection. folloing is the graph for Polynomial Regression model predection:\n")

# Visualising the Polynomial Regression results 
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('position level')
plt.ylabel('salary')
plt.show()


print("\nThe above graph shows the Polynomial Regression predection. As we can see the predection is much better "+
      "and closer to the real values. To even get the better result we can change the Degree of the regression. "+
      "Following is after changing the Degree to 4 from 2. ")


# Fitting Polynomial Regression to the dataset with differect degree
poly_reg_degree_4 = PolynomialFeatures(degree=4) # this creates 3 extra feature 
X_poly_4 = poly_reg_degree_4.fit_transform(X)
lin_reg_4 = LinearRegression()
lin_reg_4.fit(X_poly_4, y)


# Visualising the Polynomial Regression results 
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, lin_reg_4.predict(poly_reg_degree_4.fit_transform(X_grid)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('position level')
plt.ylabel('salary')
plt.show()

print("\nAs we see this time we have done even better with slightly changing the degree 2 to 4.")