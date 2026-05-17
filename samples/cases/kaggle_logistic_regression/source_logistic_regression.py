#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load the dataset
# Replace 'ex2data1.txt' with your dataset filename
data = pd.read_csv('../input/random/ex2data1.txt', header=None)
data.columns = ['Feature1', 'Feature2', 'Target']

# Features (X) and target (y)
X = data[['Feature1', 'Feature2']]
y = data['Target']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Logistic Regression model
model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# Print results
print("Model Accuracy: {:.2f}%".format(accuracy * 100))
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", class_report)

# Save the model for Kaggle
import joblib
joblib.dump(model, 'logistic_regression_model.pkl')

print("Model saved as logistic_regression_model.pkl. You can now upload it to Kaggle.")

