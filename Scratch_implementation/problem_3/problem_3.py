from problem_3.linear_regression_train import linear_regression_train
from problem_3.linear_regression_test import linear_regression_test
from problem_3.logistic_regression_train import logistic_regression_train
from problem_3.logistic_regression_test import logistic_regression_test


# Function for Problem 3
def problem_3(X_train, Y_train, X_test, Y_test):
    output_file = open('./results/problem_3.txt', "w")
    output_file.write("N, Linear reg, Logistic reg\n")

    # Train and evaluate linear regression model
    linear_reg_model = linear_regression_train(X_train, Y_train)
    lin_reg_accuracy = linear_regression_test(linear_reg_model, X_test, Y_test)

    # Train and evaluate logistic regression model
    logistic_reg_model = logistic_regression_train(X_train, Y_train)
    log_reg_accuracy = logistic_regression_test(logistic_reg_model, X_test, Y_test)

    # Save the results
    output_file.write("{:.2f}, {:.2f}\n".format(lin_reg_accuracy, log_reg_accuracy))

    output_file.close()
