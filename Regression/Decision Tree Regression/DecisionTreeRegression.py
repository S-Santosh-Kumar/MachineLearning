# Decision Tree

#Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing Dataset
Dataset = pd.read_csv('Position_Salaries.csv')
X = Dataset.iloc[:, 1:2].values 
y = Dataset.iloc[:, 2].values

#Fitting the Decision Tree regression model to the dataset
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X,y)


#Predicting the new result with Decision Tree regression model
y_pred = regressor.predict(6.5) 

#Visualizing the Decision Tree regrression results
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X,y, color = 'Red')
plt.plot(X_grid,regressor.predict(X_grid), color = 'Blue')
plt.title('Decision Tree Regression')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

