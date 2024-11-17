import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

def binary_cross_entropy_loss(Y_true, Y_pred):
    pass

def sigmoid(z):
    pass

def normalize_features(df):
    pass

def fit_logistic_regression(X, y, learning_rate, num_epochs):
    pass

def predict(X, weights, bias, threshold=0.5):
    pass

def evaluate_metrics(y_true, y_pred):
    pass

def plot_roc_curve(Y_true, Y_scores):
    pass

def plot_confusion_matrix(y_true, y_pred):
    pass

def main():
    data_train = pd.read_csv('train.csv')
    data_test = pd.read_csv('test.csv')
    
    # Normalize the features and handle categorical data
    data_train = normalize_features(data_train)
    data_test = normalize_features(data_test)
    
    # Extract features and labels
    X_train = data_train[['Gender', 'Age', 'EstimatedSalary']].values
    y_train = data_train['Purchased'].values
    X_test = data_test[['Gender', 'Age', 'EstimatedSalary']].values
    y_test = data_test['Purchased'].values

    # Train logistic regression model
    learning_rate = 0.1
    num_epochs = 10
    weights, bias = fit_logistic_regression(X_train, y_train, learning_rate, num_epochs)

    # Make predictions on the test set
    y_pred_test = predict(X_test, weights, bias)
    
    # Calculate and print evaluation metrics
    evaluate_metrics(y_test, y_pred_test)
    
    # Generate and plot ROC curve
    y_scores = sigmoid(np.dot(X_test, weights) + bias)  # Calculate probabilities for ROC
    plot_roc_curve(y_test, y_scores)
    
    # Generate and plot confusion matrix
    plot_confusion_matrix(y_test, y_pred_test)

# Run the main function
if __name__ == "__main__":
    main()
