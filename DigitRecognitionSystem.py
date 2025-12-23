import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

train_data = pd.read_csv('./Train.csv')
print("Shape of train_data: ", train_data.shape)

def one_hot_encode(y, num_classes=10):
    one_hot = np.zeros((y.shape[0], num_classes))
    one_hot[np.arange(y.shape[0]), y] = 1
    return one_hot

# Input
X = train_data.iloc[:, 1:].values # All item in row, from 2nd index to last
# Y true
y = train_data.iloc[:, 0].values # All item in row, first index only

print("Shape fo x after separating features:", X.shape)

# One-Hot encode the labels
y = one_hot_encode(y, 10)
print("One-hot encoded y: ", y.shape)
# https://stackoverflow.com/questions/49054538/how-to-split-the-data-set-without-train-test-split
# Split the data for training and testing
train_pct_index = int(0.8 * len(X))
X_train, X_test = X[:train_pct_index], X[train_pct_index:]
y_train, y_test = y[:train_pct_index], y[train_pct_index:]

# Shuffle the training data
shuffle_idx = np.random.permutation(len(X_train))
X_train = X_train[shuffle_idx]
y_train = y_train[shuffle_idx]

