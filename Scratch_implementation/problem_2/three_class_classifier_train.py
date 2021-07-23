from sklearn.linear_model import LinearRegression
from sklearn.utils import shuffle


# FUnction for training three class classifier
def three_class_classifier_train(X_train, Y_train):
    X_train, Y_train = shuffle(X_train, Y_train, random_state=0)

    linear_reg_model = LinearRegression()
    linear_reg_model.fit(X_train, Y_train)

    return linear_reg_model
