# Nutribench Project: Lightweight Models for Nutrient Estimation

This repository contains the implementation and experiments for the paper **“Lightweight Models on Nutribench”** by Erik Feng, David Bazan, Liam Yaroschuk, David Chang, and Kiran Duriseti (June 2025).
The project investigates whether small neural architectures can rival or surpass large language models (LLMs) on the **Nutribench** benchmark — a dataset that maps free-form meal descriptions to carbohydrate counts.

GitHub Repository: [https://github.com/Phangzs/nutribench-project](https://github.com/Phangzs/nutribench-project)

---

## Overview

Large models like GPT-4o or PaLM can perform surprisingly well on numeric reasoning tasks, but they come at a steep computational cost.
This project demonstrates that **compact architectures** — under 100 M parameters — can **match or outperform LLMs** on a practical domain task: estimating carb counts from natural-language food descriptions.

Through systematic tuning and empirical evaluation, we show:

* **Mean baseline**: MAE = 21.0 g
* **Linear regression (Ridge)**: MAE = 14.8 g
* **LSTM-based RNN (< 1 M params)**: MAE = 7.56 g
* **DistilBERT (< 100 M params)**: MAE = 4.44 g

These lightweight models not only reduce inference latency and energy use but also maintain reproducibility and determinism — essential for clinical or health-oriented applications.

---

## Dataset

**Nutribench**: A benchmark dataset mapping free-form text meal descriptions to carbohydrate counts (grams).
Key statistics:

| Split | Mean Length | Mean Carbs | Notes                   |
| ----- | ----------- | ---------- | ----------------------- |
| Train | 11.6 words  | 19.6 g     | Slightly smaller tails  |
| Val   | 8.8 words   | 20.2 g     | Similar distribution    |
| Test  | 18.0 words  | –          | Bimodal, longer queries |

Exploratory analysis reveals similar vocabularies across splits but a **bimodal carb distribution** and longer descriptions in the test set, explaining small generalization gaps.

> **Data availability note:** The public `data/test.csv` file currently contains placeholder rows while we wait for the official Nutribench test release.

---

## Running the Project

All experiments are orchestrated through `run_project.sh`, which wraps the individual Python scripts and keeps the workflow reproducible.

1. Create/activate the Conda environment:
   ```bash
   conda env create -f environment.yml
   conda activate nutribench
   ```
2. Use the main script:

| Command | Purpose |
| ------- | ------- |
| `./run_project.sh setup` | Install dependencies and create `data/`, `checkpoints/`, and `results/` folders. |
| `./run_project.sh data` | Build deterministic `train/val/test` CSVs (requires `data/nutribench.csv` or existing splits). |
| `./run_project.sh benchmark` | Run the mean and linear baselines in `scripts/`. |
| `./run_project.sh train transformer [--args…]` | Launch transformer fine-tuning (Optuna grid). Use `train l1`, `train l2`, or `train mean` for baselines. |
| `./run_project.sh evaluate [--split val\|test] [--checkpoint PATH]` | Run inference on the chosen split and write a CSV with queries and predicted carbs (`results/<split>_predictions.csv` by default). |
| `./run_project.sh predict [--text \"Two eggs\"]` | Load the latest checkpoint for interactive/manual predictions. |
| `./run_project.sh viz` | Regenerate the exploratory plots into `results/plots/`. |

---

## Model Architectures

### 1. Baselines

* **Mean Predictor**: Always predicts the global carb mean.
  MAE = 21.0, Accuracy@7.5 = 3.3 %.
* **Linear Regression (L1/L2)**: Tokenized bag-of-words input.
  Ridge performs best with MAE = 14.77, R² = 0.44.

### 2. LSTM-based RNN

> **Implementation status:** The RNN training/inference scripts are temporarily unavailable in this repository snapshot and will be reintroduced soon. The discussion below captures the intended architecture and findings.

* 2 stacked RNN layers + dropout regularization
* Tuned using **Optuna**:

  * Embedding = 64
  * RNN = 128 units
  * LSTM = 32 units
  * Dropout = 0.399
  * LR = 0.0017
* Training ≈ 3 min
* Validation MAE = 7.56, Accuracy@7.5 = 73.3 %

Despite its small size (< 1 M params), the RNN **beats SOTA** while remaining highly efficient and deterministic.

### 3. Encoder-only Transformer (DistilBERT)

* Models explored: `bert-base`, `roberta-base`, `distilbert-base`
* Fine-tuning with Adam optimizer and early stopping
* Grid search over:

  * Weight decay ∈ {0, 0.005, 0.01}
  * Learning rate ∈ [5e-6 → 1e-3]
* Trained on a **MacBook Pro M4 Max (128 GB RAM)**
* Best model: `distilbert-base-cased`, λ = 0.005, α = 5e-5

  * MAE = 4.44
  * Accuracy@7.5 = 86.6 %
  * Training time ≈ 6 min

---

## Performance Summary

| Model            | Parameters | MAE ↓    | Accuracy@7.5 ↑ | Training Time |
| ---------------- | ---------- | -------- | -------------- | ------------- |
| Mean Regressor   | 1          | 21.0     | 3.3 %          | –             |
| Ridge Regression | ~20 K      | 14.8     | 4.8 %          | 1 s           |
| LSTM-RNN         | < 1 M      | **7.56** | **73.3 %**     | 3 min         |
| DistilBERT       | 66 M       | **4.44** | **86.6 %**     | 6 min         |

Both RNN and DistilBERT **outperform previous LLM baselines**, while being **orders-of-magnitude smaller and faster**.

---

## Key Findings

1. **Compact models excel** when domain and numeric reasoning are well-defined.
2. **Task-specific fine-tuning** outperforms prompt-based LLM use.
3. **Distribution shift** (longer, bimodal test queries) limits generalization.
4. **Ambiguity in query wording** (“one slice”, “package”) is a persistent challenge.
5. Future improvements may involve:

   * Quantity normalization heuristics
   * Data augmentation for long-tail phrasing
   * Multi-nutrient extensions

---
<!---
## Repository Structure

```
nutribench-project/
├── data/                   # Nutribench dataset splits
├── models/
│   ├── lstm_rnn.py         # Lightweight RNN model
│   ├── transformer.py      # DistilBERT fine-tuning
│   └── baselines.py        # Mean & Linear models
├── notebooks/
│   ├── EDA.ipynb           # Exploratory data analysis
│   ├── RNN_Optuna.ipynb    # RNN hyperparameter tuning
│   └── Transformer_Search.ipynb
├── results/
│   ├── rnn_results.csv
│   ├── transformer_results.csv
│   └── plots/
├── requirements.txt
└── README.md
```

--- 

## Installation

```bash
git clone https://github.com/Phangzs/nutribench-project.git
cd nutribench-project
pip install -r requirements.txt
```

---

## Usage

### Train the RNN

```bash
python models/lstm_rnn.py --epochs 50 --batch_size 32
```

### Fine-tune DistilBERT

```bash
python models/transformer.py --model distilbert-base-cased --epochs 25
```

### Evaluate

```bash
python evaluate.py --model distilbert-base-cased
```

--- -->

## Results Visualization

Key plots from the report:

* *Top-word frequency* (p. 3): overlapping vocabularies across splits
* *Carb distribution* (p. 4): test set bimodality
* *Loss curves* (p. 9, 15): both RNN and DistilBERT converge smoothly without overfitting
* *Accuracy over epochs* (p. 10, 16): steady gains with plateau after 40 epochs

---

## Citation

If you use this code or findings, please cite:

```
@report{feng2025nutribench,
  title={Lightweight Models on Nutribench},
  author={Feng, Erik and Bazan, David and Yaroschuk, Liam and Chang, David and Duriseti, Kiran},
  year={2025},
  institution={University of California, Santa Barbara},
  month={June}
}
```

---

## License

MIT License. See `LICENSE` file for details.

---

## Acknowledgements

We thank **Andong Hua** for external test evaluation and the **UCSB ECE 180** instructional team for project guidance.
Experiments were performed on a **MacBook Pro M4 Max (128 GB RAM)** using **Optuna**, **PyTorch**, and **Hugging Face Transformers**.

---

*For full results, analyses, and figures, see the accompanying report “ECE 180 Project.pdf.”*
