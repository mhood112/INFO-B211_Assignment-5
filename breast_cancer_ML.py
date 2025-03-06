import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

try:
    # Load the breast cancer dataset
    data = load_breast_cancer()
    X = data.data # Features
    y = data.target # Target
except Exception as e:
    # If an error occurs, print the error and exit the program
    print(f"Error loading data: {e}")
    exit()

try:
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
except Exception as e:
    print(f"Error splitting data: {e}")
    exit()

try:
    # Standardize the data
    scaler = StandardScaler() 
    X_train = scaler.fit_transform(X_train) # Fit and transform the training data
    X_test = scaler.transform(X_test) # Transform the testing data
except Exception as e:
    # If an error occurs, print the error and exit the program
    print(f"Error standardizing data: {e}")
    exit()

# Initialize the models with some modifications
log_reg = LogisticRegression(random_state=42, max_iter=200)
decision_tree = DecisionTreeClassifier(random_state=42, max_depth=5)
svm = SVC(random_state=42, kernel='linear')

try:
    # Train the models
    log_reg.fit(X_train, y_train) 
    decision_tree.fit(X_train, y_train) 
    svm.fit(X_train, y_train)
except Exception as e:
    # If an error occurs, print the error and exit the program
    print(f"Error training models: {e}")
    exit()

try:
    # Make predictions
    y_pred_log_reg = log_reg.predict(X_test)
    y_pred_decision_tree = decision_tree.predict(X_test)
    y_pred_svm = svm.predict(X_test)
except Exception as e:
    # If an error occurs, print the error and exit the program
    print(f"Error making predictions: {e}")
    exit()

try:
    # Evaluate the models
    print("Logistic Regression:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred_log_reg)}")
    print(classification_report(y_test, y_pred_log_reg))

    print("Decision Tree:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred_decision_tree)}")
    print(classification_report(y_test, y_pred_decision_tree))

    print("Support Vector Machine:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred_svm)}")
    print(classification_report(y_test, y_pred_svm))
except Exception as e:
    # If an error occurs, print the error and exit the program
    print(f"Error evaluating models: {e}")
    exit()
