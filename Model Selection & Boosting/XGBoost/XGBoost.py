# XGBoost

#Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Importing Dataset
Dataset = pd.read_csv('Churn_Modelling.csv')
X = Dataset.iloc[:,3:13].values 
y = Dataset.iloc[:, [13]].values

#Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:,1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:,2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]

#Splitting the dataset into training set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state = 0)

#Fititng the XG Boost to the training set
import xgboost