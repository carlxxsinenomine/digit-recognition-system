import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.metrics as met
from scipy.special import expit
from PIL import Image

def one_hot_encode(y, num_classes=10):
    one_hot = np.zeros((y.shape[0], num_classes))
    one_hot[np.arange(y.shape[0]), y] = 1
    return one_hot


def relu(x_array):
    return np.maximum(0, x_array)


def relu_derivative(z):
    return (z > 0).astype(float)


def sigmoid(x):
    return expit(x)


def sigmoid_derivative(sigmoid_output):
    return sigmoid_output * (1 - sigmoid_output)


def softmax(Z):
    exp_scores = np.exp(Z - np.max(Z, axis=1, keepdims=True))
    return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)


def softmax_deriv_with_loss(A, y):
    return A - y


def cross_entropy(y, y_hat, batch_size):
    return -np.sum(y * np.log(y_hat + 1e-8)) / batch_size


def loss_derivative(y, y_hat):
    return y_hat - y


def confusion_matrix(y_true, y_pred):
    if y_true.ndim == 2: # Kung 2d
        y_true = np.argmax(y_true, axis=1)

    if y_pred.ndim == 2:
        y_pred = np.argmax(y_pred, axis=1)

    num_classes = 10
    matrix = np.zeros((num_classes, num_classes), dtype=int)

    for true_label, pred_label in zip(y_true, y_pred):
        matrix[true_label, pred_label] += 1

    return matrix


class NeuralNetwork:
    def __init__(self):

        self.train_data = pd.read_csv('./Train.csv')
        self.X = self.train_data.iloc[:, 1:].values / 255 # All item in row, from 2nd index to last
        # Y true
        self.y = self.train_data.iloc[:, 0].values # All item in row, first index only
        # One-Hot encode the labels
        self.y = one_hot_encode(self.y, 10)
        # https://stackoverflow.com/questions/49054538/how-to-split-the-data-set-without-train-test-split
        # Split the data for training and testing
        self.train_pct_index = int(0.8 * len(self.X))
        self.X_train, self.X_test = self.X[:self.train_pct_index], self.X[self.train_pct_index:]
        self.y_train, self.y_test = self.y[:self.train_pct_index], self.y[self.train_pct_index:]

        # Shuffle the training data
        shuffle_idx = np.random.permutation(len(self.X_train))
        self.X_train = self.X_train[shuffle_idx]
        self.y_train = self.y_train[shuffle_idx]
        # Initialize weights and biases
        # Self preference kung anong size piliin sa hidden layer
        self.__W1 = np.random.randn(784, 256) * np.sqrt(1.0 / 784)
        self.__B1 = np.zeros((1, 256))

        self.__W2 = np.random.randn(256, 128) * np.sqrt(1.0 / 256)
        self.__B2 = np.zeros((1, 128))

        self.__W3 = np.random.randn(128, 10) * np.sqrt(1.0 / 128)
        self.__B3 = np.zeros((1, 10))

        self.__Z1 = None
        self.__Z2 = None
        self.__Z3 = None

        self.__H1 = None
        self.__H2 = None
        self.__H3 = None

    def update_params(self, gradients, eta):
        w3, b3, w2, b2, w1, b1 = gradients
        self.__W3 -= w3 * eta
        self.__B3 -= b3 * eta
        self.__W2 -= w2 * eta
        self.__B2 -= b2 * eta
        self.__W1 -= w1 * eta
        self.__B1 -= b1 * eta

    def forward_pass(self, X):
        # X_train @ W1 + B1
        self.__Z1 = X @ self.__W1 + self.__B1
        # Applying sigmoid na muna kasi maya nayang ReLu HAHAHAHA sigmoid lang napagaralan ko eh
        self.__H1 = sigmoid(self.__Z1)
        # H1 @ W2 + B2
        self.__Z2 = self.__H1 @ self.__W2 + self.__B2
        #  Apply sigmoid ulit
        self.__H2 = sigmoid(self.__Z2)
        # H2 @ W3 + B3
        self.__Z3 = self.__H2 @ self.__W3 + self.__B3
        # Y pred
        self.__H3 = softmax(self.__Z3)

        return self.__H3

    """ First of all, PUTANGINANG CHAIN RULE """

    def backprop(self,X_batch, y_pred, y):
        """dL_dY_hat * dY_hat_dZ3 - small change in Z3 affects the loss"""
        dL_dZ3 = softmax_deriv_with_loss(y_pred, y)
        """ dL_dZ3 * dZ3_dW3 - small change in W3 affects the loss
         dZ3_dW3 = H2 * W3 = H2 """
        dL_dW3 = self.__H2.T @ dL_dZ3 / X_batch.shape[0]
        dL_db3 = np.sum(dL_dZ3, axis=0, keepdims=True) / X_batch.shape[0]
        """
        find deriv at the hidden neuron H2:
        Z3 = H2 * W3, then dZ3_dH2 = W3

        to compute for the dL_dZ2 = dL_dZ3 * dZ3_dH2 * dH2_dZ2

        dH2_dZ2 = sigmoid_deriv
        """
        dL_dZ2 = (dL_dZ3 @ self.__W3.T) * sigmoid_derivative(self.__H2)
        """
        Z2 = W2 * H1, so dZ2_dW2 = H1

        dL_dW2 = dL_dZ2 * dZ2_dW2
        """
        dL_dW2 = self.__H1.T @ dL_dZ2 / X_batch.shape[0]
        dL_db2 = np.sum(dL_dZ2, axis=0, keepdims=True) / X_batch.shape[0]
        """
        to find dL_dZ1 = dH1_dZ1 * dZ2_dH1 * dL_dZ2

        Z2 = H1 * W2, dZ2_dH1 = W2
        """
        dL_dZ1 = (dL_dZ2 @ self.__W2.T) * sigmoid_derivative(self.__H1)
        """
        Z1 = X_train * W1, so dZ1_dW1 = X_train
        dL_dW1 = dL_dZ1 * dZ1_dW1
        """
        dL_dW1 = X_batch.T @ dL_dZ1 / X_batch.shape[0]
        dL_db1 = np.sum(dL_dZ1, axis=0, keepdims=True) / X_batch.shape[0]

        return dL_dW3, dL_db3, dL_dW2, dL_db2, dL_dW1, dL_db1

    def evaluate(self, X, y):
        predictions = self.forward_pass(X)
        pred_classes = np.argmax(predictions, axis=1)
        true_classes = np.argmax(y, axis=1)
        accuracy = np.mean(pred_classes == true_classes)
        return accuracy

    def predict(self, X):
        # Transform to 2d
        if X.ndim == 1:
            X = X.reshape(1, -1)

        # Normalize if ever man may may val na > 1 since dapat 0 - 1 lang interval
        if X.max() > 1:
            X = X / 255.0

        probabilities = self.forward_pass(X)

        predictions = np.argmax(probabilities, axis=1)

        if len(predictions) == 1:
            return predictions[0]

        return predictions

    def show_pred(self):
        pass

    def train(self, epochs=500, batch_size=128, eta=0.1):
        # TODO: Train by batch
        n_samples = self.X_train.shape[0] # row
        n_batches = n_samples // batch_size # row size // 128

        for e in range(epochs):
            # Shuffle time
            indices = np.random.permutation(n_samples) # Better kesa sa Shuffle method
            # Use the same index para perfectly aligned parin (kaya nga mas better sa random.shuffle())
            X_shuffled = self.X_train[indices]
            y_shuffled = self.y_train[indices]

            losses = 0

            for batch in range(n_batches):
                # Starting index
                start_idx = batch * batch_size
                end_idx = start_idx + batch_size

                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]

                y_hat = self.forward_pass(X_batch)
                losses += cross_entropy(y_batch, y_hat, batch_size)

                gradients = self.backprop(X_batch, y_hat, y_batch)

                self.update_params(gradients, eta)

            train_acc = self.evaluate(self.X_train, self.y_train)
            losses_mean = losses / n_batches

            print(
                f"Epoch {e + 1}/{epochs} - "
                f"Loss: {losses_mean:.4f} - "
                f"Train Acc: {train_acc:.4f} - "
                )
            # Kasi antagal magtrain tngina
            if train_acc >= .96:
                break

        # cm = self.confusion_matrix(self.y_train, y_hats)
        # print(f"\nConfusion Matrix:\n{cm}")
        plt.imshow(self.X_test[4].reshape(28, 28), cmap='gray')
        plt.axis('off')
        plt.title(f"{self.y_test[4]}")
        plt.show()
        pred = self.predict(self.X_test[4])
        print(f"Predicted {pred}")

if __name__ == '__main__':
    nn = NeuralNetwork()
    nn.train()
# print("Shape fo x after separating features:", X.shape)


