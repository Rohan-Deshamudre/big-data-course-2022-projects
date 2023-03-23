from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
from training_set_processing import *


def get_data(df):
    # Split the data into training and test sets
    X = df[['startYear', 'runtimeMinutes', 'numVotes']]
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test


def train_SVM(df):
    X_train, X_test, y_train, y_test = get_data(df)

    # Train an SVM classifier
    clf = SVC()
    clf.fit(X_train, y_train)

    # Make predictions on the test set and evaluate the performance
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    return accuracy, y_pred


def train_logistic_regression(df):
    X_train, X_test, y_train, y_test = get_data(df)

    lr_model = LogisticRegression(max_iter=1000)
    lr_model.fit(X_train, y_train)
    lr_pred = lr_model.predict(X_test)
    lr_acc = accuracy_score(y_test, lr_pred)
    print("Logistic Regression Accuracy: {:.4f}".format(lr_acc))

    return lr_acc, lr_pred


def train_random_forest(df):
    X_train, X_test, y_train, y_test = get_data(df)

    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    rf_acc = accuracy_score(y_test, rf_pred)
    print("Random Forest Accuracy: {:.4f}".format(rf_acc))

    return rf_acc, rf_pred


def gradient_boosted_trees(df):
    X_train, X_test, y_train, y_test = get_data(df)

    gbt_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    gbt_model.fit(X_train, y_train)
    gbt_pred = gbt_model.predict(X_test)
    gbt_acc = accuracy_score(y_test, gbt_pred)
    print("Gradient Boosted Trees Accuracy: {:.4f}".format(gbt_acc))

    return gbt_acc, gbt_pred

