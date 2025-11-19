import os, random, optuna, numpy as np, torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, DataCollatorWithPadding,
    EarlyStoppingCallback, set_seed
)
import evaluate
from pathlib import Path

mae_metric = evaluate.load("mae")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.squeeze()
    mae_val = mae_metric.compute(predictions=preds, references=labels)["mae"]
    acc75 = (np.abs(preds - labels) <= 7.5).mean().item()
    return {"mae": mae_val, "accuracy@7.5": acc75}

search_space = {
    "model_name": ['bert-base-cased', 'bert-base-uncased', 'roberta-base', 'distilbert-base-uncased', 'distilbert-base-cased'],
    "weight_decay": [0, 0.005, 0.01],
    "learning_rate": [5e-4, 1e-4, 5e-5, 1e-5, 5e-6]
}
sampler = optuna.samplers.GridSampler(search_space) 
study   = optuna.create_study(direction="minimize", sampler=sampler)


def subsample_validation(dataset, frac: float, seed: int):
    if frac >= 1.0:
        return dataset
    size = int(len(dataset) * frac)
    rng  = random.Random(seed)
    idx  = rng.sample(range(len(dataset)), size)
    return dataset.select(idx)


def objective(trial: optuna.Trial):


    val_frac = 0.50                     
    tokenizer = AutoTokenizer.from_pretrained(trial.suggest_categorical(
                     "model_name", search_space["model_name"]))

    ds = load_dataset("csv",
                      data_files={"train":"data/train.csv",
                                  "validation":"data/val.csv"})
    ds = ds.map(lambda x: tokenizer(x["query"], truncation=True), batched=True)

    train_ds = ds["train"]
    eval_ds  = subsample_validation(ds["validation"], val_frac, seed=trial.number)

    model_init = lambda: AutoModelForSequenceClassification.from_pretrained(
                             trial.params["model_name"],
                             num_labels=1, problem_type="regression")

    args = TrainingArguments(
        output_dir = f"checkpoints/{trial.study.study_name}/{trial.number}",
        per_device_train_batch_size = 32,
        per_device_eval_batch_size  = 32,
        learning_rate   = trial.suggest_categorical("learning_rate", search_space["learning_rate"]),
        weight_decay    = trial.suggest_categorical("weight_decay", search_space["weight_decay"]),
        lr_scheduler_type="linear",
        num_train_epochs = 5,                    
        eval_strategy = "epoch",
        save_strategy= "epoch",
        logging_strategy     = "epoch",
        load_best_model_at_end = True,
        metric_for_best_model = "mae",
        greater_is_better     = False,
        save_total_limit      = 1,
        seed = 42,
    )

    trainer = Trainer(
        args = args,
        model_init = model_init,
        train_dataset = train_ds,
        eval_dataset  = eval_ds,
        data_collator = DataCollatorWithPadding(tokenizer),
        compute_metrics = compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )

    trainer.train()                       
    metrics = trainer.evaluate()          
    return metrics["eval_mae"]

n_trials = np.prod(list(map(len, search_space.values()))) 
study.optimize(objective, n_trials=n_trials, timeout=None, show_progress_bar=True)

print("Best MAE:",   study.best_value)
print("Best grid:",  study.best_params)
# study.trials_dataframe().to_csv("optuna_trials.csv")

