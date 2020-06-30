# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 08:31:12 2020

@author: HP
"""

#Import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas


#Import necessary functions
from accuracy import cal_rsquare
from linear_regression import simple_linear_regression

#Import the dataset 
dataset = pandas.read_csv("Data.csv")
X = dataset.iloc[:,2:20].values
y = dataset.iloc[:,1].values


#Split the dataset into the training set and test set 
from sklearn.model_selection import train_test_split 
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.25,
                                                 random_state = 0)


#Feature scaling 
"""from sklearn.preprocessing import StandardScaler
scaler_X = StandardScaler()
scaler_Y = StandardScaler()
X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.transform(X_test)
y_train = scaler_Y.fit_transform(y_train.reshape(-1,1))"""

from sklearn.linear_model import LinearRegression 
regressor = LinearRegression()
regressor = regressor.fit(X_train,y_train)
y_pred = regressor.predict(X_test)
print(cal_rsquare(y_pred,y_test))

#1.Simple linear regression (without backward elimination)
simple_linear_regression_y_pred = simple_linear_regression(X_train,X_test,
                                                           y_train)
print(f'Simple linear regression R^2: {cal_rsquare(simple_linear_regression_y_pred,y_test)}')






