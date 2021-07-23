import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from problem_4.linear_regression_train import linear_regression_train
from problem_4.linear_regression_test import linear_regression_test


# Function for Problem 4
def problem_4(X_train, Y_train, X_test, Y_test):
    degree_list = range(1, 21)
    output_file = open('./results/problem_4.txt', "w")
    output_file.write("Degree, RMSE, R2 Score\n")

    X_train = X_train[:, np.newaxis]
    X_test = X_test[:, np.newaxis]

    for degree in degree_list:
        polynomial_features = PolynomialFeatures(degree=degree)
        X_train_poly = polynomial_features.fit_transform(X_train)
        X_test_poly = polynomial_features.fit_transform(X_test)

        # Train and evaluate linear regression model
        linear_reg_model = linear_regression_train(X_train_poly, Y_train)
        RMSE, R2_score = linear_regression_test(linear_reg_model, X_test_poly, Y_test)

        # Save the results
        output_file.write("{}, {:.3f}, {:.3f}\n".format(degree, RMSE, R2_score))

    output_file.close()
