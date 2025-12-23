import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def one_hot_encode(y, num_classes=10):
    one_hot = np.zeros((y.shape[0], num_classes))
    one_hot[np.arange(y.shape[0]), y] = 1
    return one_hot


def relu(x_array):
    return np.maximum(0, x_array)


def relu_derivative(z):
    return (z > 0).astype(float)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


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


class NeuralNetwork:
    def __init__(self, learning_rate, epochs):
        self.learning_rate = learning_rate
        self.epochs = epochs

        self.train_data = pd.read_csv('./Train.csv')
        self.X = self.train_data.iloc[:, 1:].values # All item in row, from 2nd index to last
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
        self.W1 = np.random.randn(784, 256) * np.sqrt(1.0 / 784)
        self.B1 = np.zeros((1, 256))

        self.W2 = np.random.randn(256, 128) * np.sqrt(1.0 / 256)
        self.B2 = np.zeros((1, 128))

        self.W3 = np.random.randn(128, 10) * np.sqrt(1.0 / 128)
        self.B3 = np.zeros((1, 10))

        self.Z1 = None
        self.Z2 = None
        self.Z3 = None

        self.H1 = None
        self.H2 = None
        self.H3 = None

    def update_params(self, gradients, eta):
        w3, b3, w2, b2, w1, b1 = gradients
        self.W3 -= w3 * eta
        self.B3 -= b3 * eta
        self.W2 -= w2 * eta
        self.B2 -= b2 * eta
        self.W1 -= w1 * eta
        self.B1 -= b1 * eta

    def forward_pass(self,):
        # X_train @ W1 + B1
        self.Z1 = self.X_train @ self.W1 + self.B1
        # Applying sigmoid na muna kasi maya nayang ReLu HAHAHAHA sigmoid lang napagaralan ko eh
        self.H1 = sigmoid(self.Z1)
        # H1 @ W2 + B2
        self.Z2 = self.H1 @ self.W2 + self.B2
        #  Apply sigmoid ulit
        self.H2 = sigmoid(self.Z2)
        # H2 @ W3 + B3
        self.Z3 = self.H2 @ self.W3 + self.B3
        # Y pred
        self.H3 = softmax(self.Z3)

        return self. H3

    """ First of all, PUTANGINANG CHAIN RULE """

    def backprop(self,y_pred, y):
        """dL_dY_hat * dY_hat_dZ3 - small change in Z3 affects the loss"""
        dL_dZ3 = softmax_deriv_with_loss(y_pred, y)
        """ dL_dZ3 * dZ3_dW3 - small change in W3 affects the loss
         dZ3_dW3 = H2 * W3 = H2 """
        dL_dW3 = self.H2.T @ dL_dZ3 / self.X_train.shape[0]
        dL_db3 = np.sum(dL_dZ3, axis=0, keepdims=True) / self.X_train.shape[0]
        """
        find deriv at the hidden neuron H2:
        Z3 = H2 * W3, then dZ3_dH2 = W3

        to compute for the dL_dZ2 = dL_dZ3 * dZ3_dH2 * dH2_dZ2

        dH2_dZ2 = sigmoid_deriv
        """
        dL_dZ2 = (dL_dZ3 @ self.W3.T) * sigmoid_derivative(self.H2)
        """
        Z2 = W2 * H1, so dZ2_dW2 = H1

        dL_dW2 = dL_dZ2 * dZ2_dW2
        """
        dL_dW2 = self.H1.T @ dL_dZ2 / self.X_train.shape[0]
        dL_db2 = np.sum(dL_dZ2, axis=0, keepdims=True) / self.X_train.shape[0]
        """
        to find dL_dZ1 = dH1_dZ1 * dZ2_dH1 * dL_dZ2

        Z2 = H1 * W2, dZ2_dH1 = W2
        """
        dL_dZ1 = (dL_dZ2 @ self.W2.T) * sigmoid_derivative(self.H1)
        """
        Z1 = X_train * W1, so dZ1_dW1 = X_train
        dL_dW1 = dL_dZ1 * dZ1_dW1
        """
        dL_dW1 = self.X_train.T @ dL_dZ1 / self.X_train.shape[0]
        dL_db1 = np.sum(dL_dZ1, axis=0, keepdims=True) / self.X_train.shape[0]

        return dL_dW3, dL_db3, dL_dW2, dL_db2, dL_dW1, dL_db1

    def train(self):
        for e in range(self.epochs):
            y_hat = self.forward_pass()

            error = cross_entropy(self.y_train, y_hat, 33600)

            gradients = self.backprop(y_hat, self.y_train)

            self.update_params(gradients, self.learning_rate)

            pred_classes = np.argmax(y_hat, axis=1)
            true_classes = np.argmax(self.y_train, axis=1)
            accuracy = np.mean(pred_classes == true_classes)
            print(f"Epoch {e + 1}/{self.epochs} - "
                  f"Loss: {error:.4f} - "
                  f"Train Acc: {accuracy:.4f} - "
                  )

if __name__ == '__main__':
    nn = NeuralNetwork(0.1, 1000)
    nn.train()
# print("Shape fo x after separating features:", X.shape)


