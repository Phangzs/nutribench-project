import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
import numpy as np

#Load Data
train_df = pd.read_csv("train.csv")
val_df = pd.read_csv("val.csv")
test_df = pd.read_csv("test.csv")
# TF-IDF
vectorizer = TfidfVectorizer()

X_train = vectorizer.fit_transform(train_df["query"])
X_val = vectorizer.transform(val_df["query"])
X_test = vectorizer.transform(test_df["query"])

#Create the Linear Regression model
model = Ridge()
# Train the model on the training data
model.fit(X_train, train_df["carb"])

#predictions
val_pred = model.predict(X_val) #Validation set prediction
test_pred = model.predict(X_test) # Test Set prediction
train_preds = model.predict(X_train) # Training set prediction
test_df["carb"] = test_pred

# Evaluating model

y_true = val_df["carb"]


train_mae = np.mean(np.abs((train_df["carb"] - train_preds)))
val_mae = np.mean(np.abs((y_true - val_pred)))

print(f"Training MAE: {train_mae:.2f}")
print(f"Validation MAE: {val_mae:.2f}")

train_accuracy_7_5pct = 100 * np.mean(np.abs((train_df["carb"] - train_preds) / train_df["carb"]) <= 0.075)
print(f"Training 7.5% Accuracy: {train_accuracy_7_5pct:.2f}%")

val_accuracy_7_5pct = 100 * np.mean(np.abs((y_true - val_pred) / y_true) <= 0.075)
print(f"Validation 7.5% Accuracy: {val_accuracy_7_5pct:.2f}%")


#Ridge uses L2 by default
# Training MAE : 13.21
# Validation MAE: 14.77
# 7.5% Accuracy: 4.85%
