import numpy as np
from sklearn.model_selection import train_test_split

# From data
SAMPLES = 100


# Function to load data from file
def load_data(train_file):
    # Matrices to store data
    X = np.zeros(SAMPLES)
    Y = np.zeros(SAMPLES)

    # Read train data file
    file = open('./datasets/' + train_file)
    idx = 0
    for line in file:
        line_list = line.rstrip().split(' ')

        # Remove None entries from list
        line_list = list(filter(None, line_list))

        # Skipping blank lines
        if len(line_list) == 0:
            continue

        X[idx] = float(line_list[0])
        Y[idx] = float(line_list[1])
        idx += 1

    file.close()

    # Train-Test data split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

    return X_train, Y_train, X_test, Y_test
