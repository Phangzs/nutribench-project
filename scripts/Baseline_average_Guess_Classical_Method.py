import pandas as pd
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

train_df = pd.read_csv("data/train.csv")
val_df = pd.read_csv("data/val.csv")
test_df = pd.read_csv("data/test.csv")

mean_carb = train_df["label"].mean()

val_pred = [mean_carb] * len(val_df)
test_pred = [mean_carb] * len(test_df)
test_df["query"] = test_pred

train_pred = [mean_carb] * len(train_df)

train_mae = np.mean(np.abs((train_df["label"] - train_pred)))
val_mae = np.mean(np.abs((val_df["label"] - val_pred)))
train_accuracy_7_5 = 100 * np.mean(np.abs(train_df["label"] - train_pred) <= 7.5)
val_accuracy_7_5 = 100 * np.mean(np.abs(val_df["label"] - val_pred) <= 7.5)

display_metrics(
    "Mean Guess Baseline",
    train_mae,
    val_mae,
    train_accuracy_7_5,
    val_accuracy_7_5,
)


#Training MAE: 21.01
#Validation MAEL 21.08
#7.5 Accuracy: 17.35%
