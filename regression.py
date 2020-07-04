

import numpy as np

from sklearn.linear_model import LinearRegression 
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor

def linear_regression(X_train,X_test,y_train):
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


def backward_elimination(X_opt,y,significant_level):
    import statsmodels.regression.linear_model as sm 
    
    num_var = len(X_opt[0])
    while (num_var > 0):
        regressor = sm.OLS(y,X_opt).fit()
        max_Pvalue = max(regressor.pvalues).astype(float);
        if (max_Pvalue > significant_level):
            real_num_var = num_var 
            for i in range(num_var):
                prev_X_opt = X_opt 
                cur_adjusted_rvalue = regressor.rsquared_adj
                
                if (regressor.pvalues[i] == max_Pvalue):
                    X_opt = np.delete(X_opt,i,1)
                    
                    temp_regressor = sm.OLS(y,X_opt).fit()
                    new_adjusted_rvalue = temp_regressor.rsquared_adj 
                    if (new_adjusted_rvalue < cur_adjusted_rvalue):
                        return prev_X_opt
                    
                    real_num_var -= 1
            
            num_var = real_num_var
        else:
            break
    
    return X_opt


    
            
