
import numpy as np

def cal_rsquare(y_true,y_pred):
    y_avg = np.average(y_true)
    SS_res = 0
    SS_tot = 0 
    for i in range(len(y_true)):
        SS_tot += (y_true[i] - y_avg)**2
        SS_res += (y_true[i] - y_pred[i])**2
    
    return (1-SS_res/SS_tot);
