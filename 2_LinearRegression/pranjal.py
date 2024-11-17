import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def mean_squared_error(Y_true, Y_pred):
    assert len(Y_true) == len(Y_pred)
    n = len(Y_true)
    s = 0
    for i in range(n):
        s += (Y_true[i] - Y_pred[i]) ** 2
    return s / n


def fit_linear_regression(X, y):
    assert len(X) == len(y)
    X = np.insert(X, 0, 1, axis=1)
    X_T = np.transpose(X)
    return np.matmul(np.matmul(np.linalg.inv(np.matmul(X_T, X)), X_T), y)


def predict(X, weights):
    X = np.insert(X, 0, 1, axis=1)
    y = []
    for i in range(len(X)):
        assert len(X[i]) == len(weights)
        y.append(sum(X[i] * weights))
    return y


def plot_preds(X, Y_true, Y_pred):
    plt.plot(X, Y_true, "go")
    plt.plot(X, Y_pred, "ro")
    plt.show()


def normalize_features(X):
    n = len(X)
    mean = np.sum(X, axis=0) / n
    std = np.std(X, axis=0)
    return (X - mean) / std


if __name__ == "__main__":
    # Load the train data
    train_data = pd.read_csv("train.csv")

    X_train = train_data[
        ["carlength", "carwidth", "carheight", "horsepower", "peakrpm"]
    ].values
    y_train = train_data["price"].values

    test_data = pd.read_csv("test.csv")

    X_test = test_data[
        ["carlength", "carwidth", "carheight", "horsepower", "peakrpm"]
    ].values
    y_test = test_data["price"].values

    ############# Without normalized features ################
    weights = fit_linear_regression(X_train, y_train)

    y_pred = predict(X_test, weights)

    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error on Test Set: {mse}")

    plot_preds(test_data["horsepower"].values, y_test, y_pred)

    ################### Normalize features ######################
    X_train = normalize_features(X_train)

    weights = fit_linear_regression(X_train, y_train)
    X_test = normalize_features(X_test)

    y_pred = predict(X_test, weights)

    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error on Test Set: {mse}")

    plot_preds(test_data["horsepower"].values, y_test, y_pred)
