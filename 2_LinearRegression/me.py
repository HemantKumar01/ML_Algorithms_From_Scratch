# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %% [markdown]
# ## Implementing all given functions


# %%
def mean_squared_error(Y_true, Y_pred):
    assert len(Y_true) == len(Y_pred)
    squared_error = 0
    for i in range(len(Y_true)):
        squared_error += (Y_true[i] - Y_pred[i]) ** 2
    mean_squared_error = squared_error / len(Y_true)
    return mean_squared_error


# %%
def fit_linear_regression(X, y):
    A = np.random.random_sample(X.shape[1])
    b = np.random.random()
    learning_rate = 0.0009
    delta = 0
    for i in range(X.shape[0]):
        y_pred = np.dot(A, X[i]) + b
        delta = 2 * (y_pred - y[i])

        A = A - learning_rate * delta * X[i]
        b = b - learning_rate * delta

    return (A, b)


# %%
def predict(X, weights):
    A, b = weights
    y_pred = np.dot(X, A) + b
    return y_pred


# %%
def plot_preds(X, Y_true, Y_pred):
    plt.scatter(X, Y_true, color="green")
    plt.scatter(X, Y_pred, color="red")
    plt.show()


# %%
def normalize_features(X):
    mean = np.mean(X, axis=0)
    std_dev = np.std(X, axis=0)
    X_norm = (X - mean) / std_dev
    return X_norm


# %% [markdown]
# ## Running Linear Regression

# %% [markdown]
# The following codes are directly taken from sample.py given in the google drive link with datasets

# %%
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

# %%
############# Without normailzed features ################
weights = fit_linear_regression(X_train, y_train)

y_pred = predict(X_test, weights)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error on Test Set: {mse}")

plot_preds(test_data["horsepower"].values, y_test, y_pred)


# %%
################### Normalize features ######################
X_train = normalize_features(X_train)

weights = fit_linear_regression(X_train, y_train)
X_test = normalize_features(X_test)

y_pred = predict(X_test, weights)

mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error on Test Set: {mse}")

plot_preds(test_data["horsepower"].values, y_test, y_pred)
