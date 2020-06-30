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
from regression import (simple_linear_regression,polynomial_regression,
random_forest_regression)

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

#1.Simple linear regression (without backward elimination)
simple_linear_regression_y_pred = simple_linear_regression(X_train,X_test,
                                                           y_train)
print(f'Simple linear regression R^2: {cal_rsquare(y_test,simple_linear_regression_y_pred)}')

#2a.Polynomial regression (degree = 2)
degree = 2
poly_y_pred = polynomial_regression(X_train,X_test,y_train,degree)
print(f'Polynomial regression (degree = {degree}) R^2: {cal_rsquare(y_test,poly_y_pred)})')

#2b.Polynomial regression (degree = 3)
degree = 3
poly_y_pred = polynomial_regression(X_train,X_test,y_train,degree)
print(f'Polynomial regression (degree = {degree}) R^2: {cal_rsquare(y_test,poly_y_pred)})')

#3a.Random Forest Regression (num of trees = 100)
num_of_trees = 100
random_forest_y_pred = random_forest_regression(X_train,X_test,y_train,
                                                num_of_trees)
print(f'Random forest regression (trees = {num_of_trees}) R^2: {cal_rsquare(y_test,random_forest_y_pred)})')

#3b.Random Forest Regression (num of trees = 100)
num_of_trees = 200
random_forest_y_pred = random_forest_regression(X_train,X_test,y_train,
                                                num_of_trees)
print(f'Random forest regression (trees = {num_of_trees}) R^2: {cal_rsquare(y_test,random_forest_y_pred)})')

#3c.Random Forest Regression (num of trees = 100)
num_of_trees = 300
random_forest_y_pred = random_forest_regression(X_train,X_test,y_train,
                                                num_of_trees)
print(f'Random forest regression (trees = {num_of_trees}) R^2: {cal_rsquare(y_test,random_forest_y_pred)})')







