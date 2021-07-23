import numpy as np


# Tests logistic regression model
def logistic_regression_test(model, X_test, Y_test):
    pred_labels = model.predict(X_test)

    # Calculate accuracy
    classified = np.sum(pred_labels == Y_test)
    total = Y_test.shape[0]
    accuracy = classified / total

    return accuracy
