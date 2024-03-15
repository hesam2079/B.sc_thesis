from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
import xgboost as xgb
import joblib
from sklearn.metrics import classification_report

# Load data frame
df = pd.read_csv('4_emotions_data.csv')
df.dropna(inplace=True)
X = df['joined_lemmatized']
y = df['emotions']

# Encode the labels into numerical values
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, encoded_labels, test_size=0.2, random_state=42)

# Create a TF-IDF vectorizer
vectorizer = TfidfVectorizer()

# Vectorize the text data
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# SVC model
# Create SVC model
svc_model = SVC()

# Train the model
svc_model.fit(X_train_vectorized, y_train)

# Make predictions on the test set
y_pred_svc = svc_model.predict(X_test_vectorized)

# Decode the predicted labels
decoded_pred = label_encoder.inverse_transform(y_pred_svc)

# Save the model to a file
filename_svc = 'svc_model.sav'
joblib.dump(svc_model, filename_svc)

# Calculate the accuracy
report_svc = classification_report(y_test, y_pred_svc)
print("SVC report:")
print(report_svc)

# Random Forest model
# Create a Random Forest model
rf_model = RandomForestClassifier()

# Train the model
rf_model.fit(X_train_vectorized, y_train)

# Make predictions on the test set
y_pred_rf = rf_model.predict(X_test_vectorized)

# Decode the predicted labels
decoded_pred = label_encoder.inverse_transform(y_pred_rf)

# Save the model to a file
filename_rf = 'random_forest_model.sav'
joblib.dump(rf_model, filename_rf)

# Calculate the accuracy
report_rf = classification_report(y_test, y_pred_rf)
print("random forest report:")
print(report_rf)

# XGBoost model
# Create an XGBoost model
xgb_model = xgb.XGBClassifier()

# Train the model
xgb_model.fit(X_train_vectorized, y_train)

# Make predictions on the test set
y_pred_xgb = xgb_model.predict(X_test_vectorized)

# Calculate the accuracy
accuracy_xgb = accuracy_score(y_test, y_pred_xgb)

# Save the model to a file
filename_xgb = 'gradient_boosting_model.sav'
joblib.dump(xgb_model, filename_xgb)

# Calculate the accuracy
report_xgb = classification_report(y_test, y_pred_xgb)
print("xgboos report:")
print(report_xgb)

# Naive Bayes model
# Create a Naive Bayes model
nb_model = MultinomialNB()

# Train the model
nb_model.fit(X_train_vectorized, y_train)

# Make predictions on the test set
y_pred_nb = nb_model.predict(X_test_vectorized)

# Calculate the accuracy
report_nb = classification_report(y_test, y_pred_nb)

# Save the model to a file
filename_nb = 'naive_bayes_model.sav'
joblib.dump(nb_model, filename_nb)

print("naive bayes report:")
print(report_nb)