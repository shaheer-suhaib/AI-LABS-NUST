import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

lr = 0.2
epochs = 10
threshold = 0.5


df = pd.read_csv("lab10/sonar.csv")

df.iloc[:, -1] = df.iloc[:, -1].map({'R': 1, 'M': 0})

data = df.values

np.random.shuffle(data)


X = data[:, :-1]
y = data[:, -1]


split_index = int(0.8 * len(data))
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]


lr = 0.1
threshold = 0.5
bias = 0
features = X_train.shape[1]
W = np.zeros(features)


while True:
    correct = 0
    for i in range(len(X_train)):
        output = np.dot(X_train[i], W) + bias
        output = 1 if output >= threshold else 0
        error = y_train[i] - output

        if error == 0:
            correct += 1
        else:
            for j in range(features):
                W[j] = W[j] + lr * error * X_train[i, j]
            bias = bias + lr * error

    train_accuracy = correct / len(X_train) * 100
    print("train accuracy:", train_accuracy)

    if train_accuracy >= 75:
        break


correct_test = 0
predictions = np.zeros(len(X_test), dtype=int)

for i in range(len(X_test)):
    output = np.dot(X_test[i], W) + bias
    output = 1 if output >= threshold else 0
    predictions[i] = output
    if output == y_test[i]:
        correct_test += 1

test_accuracy = correct_test / len(X_test) * 100
print("test accuracy:", test_accuracy)

cm = confusion_matrix(y_test.astype(int), predictions)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()