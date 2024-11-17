import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def mean_squared_error(Y_true, Y_pred):
    pass


def fit_linear_regression(X, y):
    pass


def predict(X, weights):
    pass


def plot_preds(X, Y_true, Y_pred):
    pass


def normalize_features(X):
    pass


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

    ############# Without normailzed features ################
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
