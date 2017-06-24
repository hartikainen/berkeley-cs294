import json
import pickle

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

EPS = 1e-8 # epsilon constant for numeric stability



AVAILABLE_ENVS = (
    'Ant-v1',
    'HalfCheetah-v1',
    'Hopper-v1',
    'Humanoid-v1',
    'Reacher-v1',
    'Walker2d-v1'
)

LOG_LEVELS = (
    tf.logging.ERROR,
    tf.logging.INFO,
    tf.logging.WARN,
    tf.logging.ERROR,
    tf.logging.FATAL
)


def dump_results(results_file, results):
    data = []
    try:
        with open(results_file, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        pass
    finally:
        data.append(results)
        with open(results_file, "w") as f:
            json.dump(data, f, sort_keys=True,
                      indent=4, separators=(',', ': '))


def train_test_val_split(X, y,
                         train_prop, val_prop, test_prop, N_dev=0,
                         verbose=True):
    """ Split the dataset (X, y) into train, validation, and test sets.

    Also, possibly create a small dataset for development purposes. Note that
    the proportions should sum to 1.
    Arguments:
    X -- Feature matrix, shape of (N, D_in)
    y -- Labels, shape of (N, num_classes)
    train_prop -- Proportion of training data
    val_prop -- Proportion of validation data
    test_prop -- Proportion of test data
    N_dev -- Number of examples to include in the dev data
    """
    N_total = X.shape[0]
    N_train = int(np.floor(N_total * train_prop))
    N_val   = int(np.ceil(N_total * val_prop))
    N_test  = N_total - N_train - N_val
    N_dev   = min(N_dev, N_total)

    assert(N_train + N_val + N_test == N_total)

    # Split the data into test set and temporary set, which will be
    # split into training and validation sets
    X_tmp, X_test, y_tmp, y_test = train_test_split(X,
                                                    y,
                                                    test_size=N_test)

    # Split X_tmp into training and validation
    X_train, X_val, y_train, y_val = train_test_split(X_tmp,
                                                      y_tmp,
                                                      train_size=N_train)

    # Create the development set, which is just a small subset of
    # the training set.
    mask = np.random.choice(N_train, N_dev, replace=False)
    X_dev = X_train[mask]
    y_dev = y_train[mask]

    if verbose:
        print('Train data: ', X_train.shape, X_train.dtype)
        assert(X_train.shape[0] == N_train)
        print('Train labels: ', y_train.shape, y_train.dtype)
        assert(y_train.shape[0] == N_train)

        print('Validation data: ', X_val.shape, X_val.dtype)
        assert(X_val.shape[0] == N_val)
        print('Validation labels: ', y_val.shape, y_val.dtype)
        assert(y_val.shape[0] == N_val)

        print('Test data: ', X_test.shape, X_test.dtype)
        assert(X_test.shape[0] == N_test)
        print('Test labels: ', y_test.shape, y_test.dtype)
        assert(y_test.shape[0] == N_test)

        print('Dev data: ', X_dev.shape, X_dev.dtype)
        assert(X_dev.shape[0] == N_dev)
        print('Dev labels: ', y_dev.shape, y_dev.dtype)
        assert(y_dev.shape[0] == N_dev)

    data = {
        "X_train": X_train, "y_train": y_train,
        "X_val": X_val, "y_val": y_val,
        "X_test": X_test, "y_test": y_test,
        "X_dev": X_dev, "y_dev": y_dev
    }

    return data

def load_expert_data(filename, verbose=False):
    """ Load the expert data from pickle saved in filename"""

    expert_data = None
    with open(filename, "rb") as f:
        expert_data = pickle.load(f)

    observations = expert_data["observations"].astype('float32')
    actions = np.squeeze(expert_data["actions"].astype('float32'))

    if verbose:
        # As a sanity check, print out the size of the training and test data.
        print('observations shape: ', observations.shape)
        print('actions shape: ', actions.shape)

    return observations, actions


