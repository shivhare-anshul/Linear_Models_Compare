from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle


# Trains logistic regression model
def logistic_regression_train(X_train, Y_train):
    X_train, Y_train = shuffle(X_train, Y_train, random_state=0)

    logistic_reg_model = LogisticRegression(max_iter=750)
    logistic_reg_model.fit(X_train, Y_train)

    return logistic_reg_model
