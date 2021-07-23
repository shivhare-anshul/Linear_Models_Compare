import random
import numpy as np
from problem_1.linear_regression_train import linear_regression_train
from problem_1.linear_regression_test import linear_regression_test
from problem_1.logistic_regression_train import logistic_regression_train
from problem_1.logistic_regression_test import logistic_regression_test


# Function for Problem 1c
def problem_1c(X_train, Y_train, X_test, Y_test):
    N_list = [10, 20, 50, 100, 250, 500, 1000, 1500, 2000]
    output_file = open('./results/problem_1c.txt', "w")
    output_file.write("N, Linear reg, Logistic reg\n")

    # Setting class label of -1 to 0
    Y_train[Y_train == -1] = 0
    Y_test[Y_test == -1] = 0

    # Separate two classes
    X_class_0 = X_train[Y_train == 0]
    Y_class_0 = Y_train[Y_train == 0]
    X_class_1 = X_train[Y_train == 1]
    Y_class_1 = Y_train[Y_train == 1]

    for N in N_list:
        rand_list = random.sample(range(0, 1000), N // 2)

        # Sample N/2 data points from each class
        X_0 = X_class_0[rand_list]
        Y_0 = Y_class_0[rand_list]
        X_1 = X_class_1[rand_list]
        Y_1 = Y_class_1[rand_list]

        # Create train dataset
        X_train_N = np.concatenate([X_0, X_1])
        Y_train_N = np.concatenate([Y_0, Y_1])

        # Train and evaluate linear and logistic regression models
        linear_reg_model = linear_regression_train(X_train_N, Y_train_N)
        lin_reg_accuracy = linear_regression_test(linear_reg_model, X_test, Y_test)

        logistic_reg_model = logistic_regression_train(X_train_N, Y_train_N)
        log_reg_accuracy = logistic_regression_test(logistic_reg_model, X_test, Y_test)

        # Save the results
        output_file.write("{}, {:.2f}, {:.2f}\n".format(N, lin_reg_accuracy, log_reg_accuracy))

    output_file.close()
