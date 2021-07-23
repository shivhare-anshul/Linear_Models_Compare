import numpy as np


# Tests one_vs_rest models
def one_vs_rest_test(models, X_test, Y_test):
    NUM_CLASSES = len(models)
    [model_1, model_2, model_3] = models
    pred_labels = np.zeros([NUM_CLASSES, Y_test.shape[0]])

    # Predict class 1 vs class 2, 3 model
    pred_model_1 = model_1.predict(X_test)
    pred_labels[0] = pred_model_1

    # Predict class 2 vs class 1, 3 model
    pred_model_2 = model_2.predict(X_test)
    pred_labels[1] = pred_model_2

    # Predict class 3 vs class 1, 2 model
    pred_model_3 = model_3.predict(X_test)
    pred_labels[2] = pred_model_3

    # Choose the class with lowest regression output
    pred_labels = np.argmin(pred_labels, axis=0) + 1

    # Calculate accuracy
    classified = np.sum(pred_labels == Y_test)
    total = Y_test.shape[0]
    accuracy = classified / total

    return accuracy
