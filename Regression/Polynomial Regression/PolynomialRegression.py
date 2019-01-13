# Polynomial Regression
#Importing Libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing Dataset
Dataset = pd.read_csv('Position_Salaries.csv')
X = Dataset.iloc[:, 1:2].values 
y = Dataset.iloc[:, 2].values

#Fitting the linear regression to the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,y) 

#Fitting the polynomial regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 5)
X_poly = poly_reg.fit_transform(X)

lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

#Visualizing the linear regression results
plt.scatter(X,y, color = 'Red')
plt.plot(X,lin_reg.predict(X), color = 'Blue')
plt.title('Linear Regression')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

#Visualizing the Polynomial regrression results
X_grid = np.arange(min(X), max(X),0.1)
X_grid = X_grid.reshape(len(X_grid),1)
plt.scatter(X,y, color = 'Red')
plt.plot(X_grid,lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color = 'Blue')
plt.title('Polynomial Regression')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()