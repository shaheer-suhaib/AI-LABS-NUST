import numpy as np
import pandas as pd


df = pd.read_csv("lab10/Iris.csv")


df["Species"] = df["Species"].map({"Iris-setosa": 1, 
                                   "Iris-versicolor": 0,
                                   "Iris-virginica": 0})

data = df.values


np.random.shuffle(data)


X = data[:, 1:-1]    
y = data[:, -1]


split_index = int(0.8 * len(data))
X_train = X[:split_index]
y_train = y[:split_index]
X_test = X[split_index:]
y_test = y[split_index:]


lr = 0.1
epochs = 10000
threshold = 0.5

input_size = X_train.shape[1]
hidden_size = 4
output_size = 1


W1 = np.zeros((input_size, hidden_size))
b1 = np.zeros(hidden_size)
W2 = np.zeros((hidden_size, output_size))
b2 = np.zeros(output_size)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def d_sigmoid(x):
    return x * (1 - x)

for ep in range(epochs):

    correct = 0

    for i in range(len(X_train)):

    
        Z1 = np.dot(X_train[i], W1) + b1
        A1 = sigmoid(Z1)
        Z2 = np.dot(A1, W2) + b2
        A2 = sigmoid(Z2)

        # prediction check
        pred = 1 if A2 >= threshold else 0
        if pred == y_train[i]:
            correct += 1

        # error
        error = A2 - y_train[i]

        # backprop
        delta2 = error * d_sigmoid(A2)
        delta1 = np.dot(W2, delta2) * d_sigmoid(A1)

        # gradients
        grad_W2 = np.outer(A1, delta2)
        grad_b2 = delta2
        grad_W1 = np.outer(X_train[i], delta1)
        grad_b1 = delta1

     
        W2 = W2 - lr * grad_W2
        b2 = b2 - lr * grad_b2
        W1 = W1 - lr * grad_W1
        b1 = b1 - lr * grad_b1

    if ep % 1000 == 0:
        acc = (correct / len(X_train)) * 100
        print("epoch:", ep, "training accuracy:", acc)


train_accuracy = (correct / len(X_train)) * 100
print("final training accuracy:", train_accuracy)


correct_test = 0

for i in range(len(X_test)):
    Z1 = np.dot(X_test[i], W1) + b1
    A1 = sigmoid(Z1)
    Z2 = np.dot(A1, W2) + b2
    A2 = sigmoid(Z2)
    pred = 1 if A2 >= threshold else 0
    if pred == y_test[i]:
        correct_test += 1

test_accuracy = (correct_test / len(X_test)) * 100
print("testing accuracy:", test_accuracy)
