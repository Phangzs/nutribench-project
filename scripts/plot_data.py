import os
from pathlib import Path
from collections import Counter
import re

OUTPUT_DIR = Path("results/plots")
CACHE_ROOT = OUTPUT_DIR / ".cache"
MPL_CACHE_DIR = CACHE_ROOT / "matplotlib"
FONT_CACHE_DIR = CACHE_ROOT / "fontconfig"
for path in (MPL_CACHE_DIR, FONT_CACHE_DIR):
    path.mkdir(parents=True, exist_ok=True)

os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_ROOT))

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def save_figure(fig, filename):
    """Persist the figure for headless runs and close it to free memory."""
    fig.tight_layout()
    out_path = OUTPUT_DIR / filename
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"[plot_data] Saved {out_path}")

# Load datasets
train = pd.read_csv("data/train.csv")
val = pd.read_csv("data/val.csv")
test = pd.read_csv("data/test.csv")

# --- 1. Carbs distribution ---


# compute percentiles over both splits
combined = pd.concat([train['label'], val['label']])
low, high = combined.quantile([0.01, 0.99])

fig, ax = plt.subplots(figsize=(8, 4))
sns.kdeplot(train['label'], label='Train', fill=True, ax=ax)
sns.kdeplot(val['label'], label='Validation', fill=True, ax=ax)
ax.set_title("Carbs Distribution (1st–99th percentile)")
ax.set_xlabel("Carbs (g)")
ax.set_xlim(low, high)       # drop the top/bottom 1%
ax.legend()
save_figure(fig, "carb_distribution.png")

# --- 2. Description word lengths ---
def word_length_series(df): return df['query'].str.split().apply(len)

fig, ax = plt.subplots(figsize=(8, 4))
for name, df in [('Train', train), ('Val', val), ('Test', test)]:
    sns.histplot(word_length_series(df), bins=30, kde=True, stat='density', label=name, ax=ax)
ax.set_title("Description Word Count Distribution")
ax.set_xlabel("# Words per Query")
ax.legend()
save_figure(fig, "word_count_distribution.png")

# --- 3. Top 20 most common words in train ---
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
stop_words = set(ENGLISH_STOP_WORDS)




def clean_and_tokenize(texts):
    words = []
    for text in texts:
        tokens = re.findall(r"\b\w+\b", text.lower())
        # drop stop-words _and_ any 1-letter token
        filtered = [t for t in tokens
                    if t not in stop_words
                       and len(t) > 1]
        words.extend(filtered)
    return words

train_words = clean_and_tokenize(test['query'])
word_counts = Counter(train_words).most_common(20)
words, counts = zip(*word_counts)          # was: words, freqs = zip(*word_counts)
total_words   = len(train_words)           # NEW – denominator
freqs         = [c / total_words for c in counts]   # NEW – relative frequencies


fig, ax = plt.subplots(figsize=(10, 4))
color = sns.color_palette("mako", n_colors=1)[0]
sns.barplot(x=list(words), y=list(freqs), color=color, ax=ax)
ax.set_title("Top 20 Most Frequent Words (Test)")
for label in ax.get_xticklabels():
    label.set_rotation(45)
    label.set_horizontalalignment("right")
ax.set_ylabel("Frequency")
save_figure(fig, "top_words.png")
