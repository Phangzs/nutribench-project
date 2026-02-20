import random, optuna, numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, DataCollatorWithPadding,
    EarlyStoppingCallback
)
import evaluate

mae_metric = evaluate.load("mae")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.squeeze()
    mae_val = mae_metric.compute(predictions=preds, references=labels)["mae"]
    acc75 = (np.abs(preds - labels) <= 7.5).mean().item()
    return {"mae": mae_val, "accuracy@7.5": acc75}

search_space = {
    "model_name": [
        "bert-base-cased",
        "bert-base-uncased",
        "roberta-base",
        "distilbert-base-uncased",
        "distilbert-base-cased",
    ],
    "weight_decay": [0, 0.005, 0.01],
    "learning_rate": [1e-3, 5e-4, 1e-4, 5e-5, 1e-5, 5e-6],
}
sampler = optuna.samplers.GridSampler(search_space) 
study   = optuna.create_study(direction="minimize", sampler=sampler)


class ResamplingTrainer(Trainer):
    """Trainer that re-samples validation each epoch to drop a random half."""
    def __init__(self, *args, val_frac: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.val_frac = val_frac
        self._full_eval_dataset = self.eval_dataset

    def get_eval_dataloader(self, eval_dataset=None):
        base_ds = eval_dataset or self._full_eval_dataset
        if base_ds is None:
            return super().get_eval_dataloader(eval_dataset)
        if self.val_frac < 1.0:
            size = max(1, int(len(base_ds) * self.val_frac))
            epoch_int = int(self.state.epoch or 0)
            rng = random.Random((self.args.seed or 0) + epoch_int)
            sampled = base_ds.select(rng.sample(range(len(base_ds)), size))
            return super().get_eval_dataloader(sampled)
        return super().get_eval_dataloader(base_ds)


def objective(trial: optuna.Trial):


    val_frac = 0.5  # randomly omit half of validation each epoch                    
    tokenizer = AutoTokenizer.from_pretrained(trial.suggest_categorical(
                     "model_name", search_space["model_name"]))

    ds = load_dataset("csv",
                      data_files={"train":"data/train.csv",
                                  "validation":"data/val.csv"})
    ds = ds.map(lambda x: tokenizer(x["query"], truncation=True), batched=True)

    train_ds = ds["train"]
    eval_ds  = ds["validation"]

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
        num_train_epochs = 50,                    
        eval_strategy = "epoch",
        save_strategy= "epoch",
        logging_strategy     = "epoch",
        load_best_model_at_end = True,
        metric_for_best_model = "mae",
        greater_is_better     = False,
        save_total_limit      = 1, # TODO: should save every 500 or something
        seed = 42,
    )

    trainer = ResamplingTrainer(
        args = args,
        model_init = model_init,
        train_dataset = train_ds,
        eval_dataset  = eval_ds,
        data_collator = DataCollatorWithPadding(tokenizer),
        compute_metrics = compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
        val_frac=val_frac,
    )

    trainer.train()                       
    metrics = trainer.evaluate()          
    return metrics["eval_mae"]

n_trials = np.prod(list(map(len, search_space.values()))) 
study.optimize(objective, n_trials=n_trials, timeout=None, show_progress_bar=True)

print("Best MAE:",   study.best_value)
print("Best grid:",  study.best_params)
study.trials_dataframe().to_csv("optuna_trials.csv")
