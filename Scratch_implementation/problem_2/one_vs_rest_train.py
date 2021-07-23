import numpy as np
from sklearn.linear_model import LinearRegression


# Function for training one vs rest classifiers
def one_vs_rest_train(X_train, Y_train):
    #########################################################################
    # Train 1 vs rest classifier
    X_train_mod = X_train.copy()
    Y_train_mod = Y_train.copy()

    # Set class 1 with label 0 and class 2 and class 3 with a single label 1
    Y_train_mod[Y_train == 1] = 0
    Y_train_mod[Y_train == 2] = 1
    Y_train_mod[Y_train == 3] = 1

    model_1 = LinearRegression()
    model_1.fit(X_train_mod, Y_train_mod)
    #########################################################################
    # Train 2 vs rest classifier
    X_train_mod = X_train.copy()
    Y_train_mod = Y_train.copy()

    # Set class 2 with label 0 and class 1 and class 3 with a single label 1
    Y_train_mod[Y_train == 1] = 1
    Y_train_mod[Y_train == 2] = 0
    Y_train_mod[Y_train == 3] = 1

    model_2 = LinearRegression()
    model_2.fit(X_train_mod, Y_train_mod)
    #########################################################################
    # Train 3 vs rest classifier
    X_train_mod = X_train.copy()
    Y_train_mod = Y_train.copy()

    # Set class 3 with label 0 and class 1 and class 2 with a single label 1
    Y_train_mod[Y_train == 1] = 1
    Y_train_mod[Y_train == 2] = 1
    Y_train_mod[Y_train == 3] = 0

    model_3 = LinearRegression()
    model_3.fit(X_train_mod, Y_train_mod)
    #########################################################################

    return [model_1, model_2, model_3]
