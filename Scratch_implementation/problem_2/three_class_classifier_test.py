import numpy as np


# Tests three_class_classifier regression model
def three_class_classifier_test(model, X_test, Y_test):
    pred_labels = model.predict(X_test)

    # Convert real values to class labels
    pred_labels = np.round(pred_labels)

    # Calculate accuracy
    classified = np.sum(pred_labels == Y_test)
    total = Y_test.shape[0]
    accuracy = classified / total

    return accuracy
