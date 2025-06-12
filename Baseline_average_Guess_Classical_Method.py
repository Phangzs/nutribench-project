import pandas as pd
import numpy as np



train_df = pd.read_csv("train.csv")
val_df = pd.read_csv("val.csv")
test_df = pd.read_csv("test.csv")

mean_carb = train_df["carb"].mean()

val_pred = [mean_carb] * len(val_df)
test_pred = [mean_carb] * len(test_df)
test_df["carb"] = test_pred

train_pred = [mean_carb] * len(train_df)

train_mae = np.mean(np.abs((train_df["carb"] - train_pred)))
val_mae = np.mean(np.abs((val_df["carb"] - val_pred)))

print(f"Training MAE: {train_mae:.2f}")
print(f"Validation MAE: {val_mae:.2f}")

train_accuracy_7_5pct = 100 * np.mean(np.abs((train_df["carb"] - train_pred) / train_df["carb"]) <= 0.075)

print(f"Training 7.5% Accuracy: {train_accuracy_7_5pct:.2f}%")

val_accuracy_7_5pct = 100 * np.mean(np.abs((val_df["carb"] - val_pred) / val_df["carb"]) <= 0.075)
print(f"Validation 7.5% Accuracy: {val_accuracy_7_5pct:.2f}%")


#Training MAE: 21.01
#Validation MAEL 21.08
#7.5% Accuracy: 3.30%