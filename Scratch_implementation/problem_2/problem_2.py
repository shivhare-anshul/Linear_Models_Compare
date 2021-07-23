from problem_2.one_vs_rest_train import one_vs_rest_train
from problem_2.one_vs_rest_test import one_vs_rest_test
from problem_2.three_class_classifier_train import three_class_classifier_train
from problem_2.three_class_classifier_test import three_class_classifier_test


# Function for Problem 2
def problem_2(X_train, Y_train, X_test, Y_test):
    output_file = open('./results/problem_2.txt', "w")
    output_file.write("One vs Rest, 3 class regression\n")

    # Train and evaluate one vs rest regression models
    models = one_vs_rest_train(X_train, Y_train)
    one_vs_rest_accuracy = one_vs_rest_test(models, X_test, Y_test)

    # Train and evaluate three class regression models
    model = three_class_classifier_train(X_train, Y_train)
    three_class_classifier_accuracy = three_class_classifier_test(model, X_test, Y_test)

    # Save the results
    output_file.write("{:.2f}, {:.2f}\n".format(one_vs_rest_accuracy, three_class_classifier_accuracy))

    output_file.close()
