import numpy as np

# From data
TRAIN_SAMPLES = 2000
TEST_SAMPLES = 1000


# Function to load data from file
def load_data(train_file, test_file, DIMENSIONS):
    # Matrices to store data
    X_train = np.zeros([TRAIN_SAMPLES, DIMENSIONS])
    X_test = np.zeros([TEST_SAMPLES, DIMENSIONS])
    Y_train = np.zeros(TRAIN_SAMPLES)
    Y_test = np.zeros(TEST_SAMPLES)

    # Read train data file
    file = open('./datasets/' + train_file)
    idx = 0
    for line in file:
        line_list = line.rstrip().split(',')

        # Remove None entries from list
        line_list = list(filter(None, line_list))

        # Skipping blank lines
        if len(line_list) == 0:
            continue

        X_train[idx] = np.array(line_list[:-1], dtype=float)
        Y_train[idx] = int(line_list[-1])
        idx += 1

    file.close()

    # Read test data file
    file = open('./datasets/' + test_file)
    idx = 0
    for line in file:
        line_list = line.rstrip().split(',')

        # Remove None entries from list
        line_list = list(filter(None, line_list))

        # Skipping blank lines
        if len(line_list) == 0:
            continue

        X_test[idx] = np.array(line_list[:-1], dtype=float)
        Y_test[idx] = int(line_list[-1])
        idx += 1

    file.close()

    return X_train, Y_train, X_test, Y_test
