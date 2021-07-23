import os
from problem_1.load_data import load_data
from problem_1.problem_1a import problem_1a
from problem_1.problem_1b import problem_1b
from problem_1.problem_1c import problem_1c


# Create results directory
os.makedirs('results', exist_ok=True)

# Problem 1a
X_train, Y_train, X_test, Y_test = load_data('Gamma_train.txt', 'Gamma_test.txt', 2)
problem_1a(X_train, Y_train, X_test, Y_test)

# Problem 1b
X_train, Y_train, X_test, Y_test = load_data('Uniform_train.txt', 'Uniform_test.txt', 2)
problem_1b(X_train, Y_train, X_test, Y_test)

# Problem 1c
X_train, Y_train, X_test, Y_test = load_data('Normal_train_10D.txt', 'Normal_test_10D.txt', 10)
problem_1c(X_train, Y_train, X_test, Y_test)
