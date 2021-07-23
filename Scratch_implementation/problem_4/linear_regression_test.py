import numpy as np
from sklearn.metrics import mean_squared_error, r2_score


# Tests linear regression model
def linear_regression_test(model, X_test, Y_test):
    pred_values = model.predict(X_test)

    # Calculate Root Mean Square Error and R2 score
    RMSE = np.sqrt(mean_squared_error(Y_test, pred_values))
    R2_score = r2_score(Y_test, pred_values)

    return RMSE, R2_score
