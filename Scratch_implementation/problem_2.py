import os
from problem_2.load_data import load_data
from problem_2.problem_2 import problem_2


# Create results directory
os.makedirs('results', exist_ok=True)

# Problem 2
X_train, Y_train, X_test, Y_test = load_data('iris_dataset.txt')
problem_2(X_train, Y_train, X_test, Y_test)
