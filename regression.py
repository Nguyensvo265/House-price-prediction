# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 21:23:20 2020

@author: HP
"""

from sklearn.linear_model import LinearRegression 
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor

def simple_linear_regression(X_train,X_test,y_train):
    regressor = LinearRegression()
    regressor = regressor.fit(X_train,y_train)
    y_pred = regressor.predict(X_test)
    return y_pred


def polynomial_regression(X_train,X_test,y_train,degree):
    poly = PolynomialFeatures(degree = degree)
    X_poly = poly.fit_transform(X_train)
    
    pol_regressor = LinearRegression()
    pol_regressor = pol_regressor.fit(X_poly,y_train)
    
    y_pred = pol_regressor.predict(poly.transform(X_test))
    return y_pred 


def random_forest_regression(X_train,X_test,y_train,num_of_trees):
    regressor = RandomForestRegressor(num_of_trees,random_state = 0)
    regressor = regressor.fit(X_train,y_train)
    y_pred = regressor.predict(X_test)
    return y_pred 
