import pandas as pd
import numpy as np                                                                                  

data = pd.read_csv("diabetes.csv")


X = data.drop("Outcome", axis=1).values
y = data["Outcome"].values

X

k = 2
fold_size = len(data) // k

accuracies = []

for fold in range(k):

    print(f"\n=========== FOLD {fold+1} ===========")

    if fold == 0:
        X_train = X[:fold_size]
        y_train = y[:fold_size]
        X_test  = X[fold_size:]
        y_test  = y[fold_size:]
    else:
        X_test  = X[:fold_size]
        y_test  = y[:fold_size]
        X_train = X[fold_size:]
        y_train = y[fold_size:]

  
    train_data = {}
    for i in range(len(X_train)):
        train_data[tuple(X_train[i])] = y_train[i]

    # classify 
    correct = 0

    for i in range(len(X_test)):
        pred = naive_bayes(train_data, [0, 1], X_test[i])
        if pred == y_test[i]:
            correct += 1

    accuracy = correct / len(X_test)
    accuracies.append(accuracy)

    print(f"Fold Accuracy = {accuracy:.4f}")

print("===============================")
print(f"Final Accuracy = {np.mean(accuracies):.4f}")
