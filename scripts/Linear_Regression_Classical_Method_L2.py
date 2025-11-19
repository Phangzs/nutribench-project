import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
import numpy as np


def display_metrics(model_name, train_mae, val_mae, train_acc, val_acc):
    """Pretty-print metrics so each script reports results uniformly."""
    border = "=" * 60
    print(f"\n{border}")
    print(f"{model_name:^60}")
    print(border)
    print(f"{'Split':<15}{'MAE':>12}{'Accuracy@7.5':>20}")
    print("-" * 60)
    print(f"{'Train':<15}{train_mae:>12.2f}{train_acc:>19.2f}%")
    print(f"{'Validation':<15}{val_mae:>12.2f}{val_acc:>19.2f}%")
    print(border)

#Load Data
train_df = pd.read_csv("data/train.csv")
val_df = pd.read_csv("data/val.csv")
test_df = pd.read_csv("data/test.csv")
# TF-IDF
vectorizer = TfidfVectorizer()

X_train = vectorizer.fit_transform(train_df["query"])
X_val = vectorizer.transform(val_df["query"])
X_test = vectorizer.transform(test_df["query"])

#Create the Linear Regression model
model = Ridge()
# Train the model on the training data
model.fit(X_train, train_df["label"])

#predictions
val_pred = model.predict(X_val) #Validation set prediction
test_pred = model.predict(X_test) # Test Set prediction
train_preds = model.predict(X_train) # Training set prediction
test_df["query"] = test_pred

# Evaluating model

y_true = val_df["label"]


train_mae = np.mean(np.abs((train_df["label"] - train_preds)))
val_mae = np.mean(np.abs((y_true - val_pred)))
train_accuracy_7_5 = 100 * np.mean(np.abs(train_df["label"] - train_preds) <= 7.5)
val_accuracy_7_5 = 100 * np.mean(np.abs(y_true - val_pred) <= 7.5)

display_metrics(
    "Ridge Regression (L2)",
    train_mae,
    val_mae,
    train_accuracy_7_5,
    val_accuracy_7_5,
)

# Ridge uses L2 by default
# Training MAE : 13.21
# Validation MAE: 14.77
# Accuracy @ 7.5: 46.90%
