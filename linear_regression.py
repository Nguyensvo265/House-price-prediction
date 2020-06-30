# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 21:23:20 2020

@author: HP
"""

from sklearn.linear_model import LinearRegression 
def simple_linear_regression(X_train,X_test,y_train):
    regressor = LinearRegression()
    regressor = regressor.fit(X_train,y_train)
    y_pred = regressor.predict(X_test)
    return y_pred