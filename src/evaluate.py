from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

def calculate_rmse(y_test, y_pred):
    return np.sqrt(mean_squared_error(y_test, y_pred))

def calculate_r2(y_test, y_pred):
    return r2_score(y_test, y_pred)