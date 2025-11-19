#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="$PROJECT_ROOT/data"
CHECKPOINT_DIR="$PROJECT_ROOT/checkpoints"
RESULTS_DIR="$PROJECT_ROOT/results"
LOG_DIR="$PROJECT_ROOT/logs"
SCRIPT_DIR="$PROJECT_ROOT/scripts"

usage() {
    cat <<'EOF'
Usage: ./run_project.sh <mode> [options]

Modes
  setup           Install dependencies (conda env update) and create folders.
  data            Create deterministic train/val/test splits from data/*.csv.
  benchmark       Run the mean, L1, and L2 baselines.
  train [target]  Train a model: transformer (default), mean, l1, or l2.
  evaluate        Run inference on a split and save predictions to CSV.
  predict         Interactive inference with a saved checkpoint.
  viz             Generate exploratory data plots.

Examples
  ./run_project.sh setup
  ./run_project.sh data
  ./run_project.sh benchmark
  ./run_project.sh train transformer
  ./run_project.sh evaluate --split test
  ./run_project.sh predict --checkpoint checkpoints/study/0/checkpoint-500
EOF
}

log() {
    echo "[run_project] $*"
}

ensure_structure() {
    mkdir -p "$DATA_DIR" "$CHECKPOINT_DIR" "$RESULTS_DIR" "$LOG_DIR"
}

require_data_splits() {
    local missing=0
    for split in train val test; do
        if [[ ! -f "$DATA_DIR/$split.csv" ]]; then
            log "Missing data split: $DATA_DIR/$split.csv"
            missing=1
        fi
    done
    if [[ $missing -eq 1 ]]; then
        log "Run './run_project.sh data' to generate the splits."
        exit 1
    fi
}

find_latest_checkpoint() {
    if [[ ! -d "$CHECKPOINT_DIR" ]]; then
        return 1
    fi
    python - "$CHECKPOINT_DIR" <<'PY'
import os
import sys

root = sys.argv[1]
latest_path = None
latest_mtime = -1.0
model_files = ("pytorch_model.bin", "model.safetensors")

for current, _, files in os.walk(root):
    available = [f for f in model_files if f in files]
    if not available:
        continue
    # prefer pytorch bin when both exist
    target = "pytorch_model.bin" if "pytorch_model.bin" in available else available[0]
    mtime = os.path.getmtime(os.path.join(current, target))
    if mtime > latest_mtime:
        latest_mtime = mtime
        latest_path = current

if latest_path:
    print(latest_path)
PY
}

setup_env() {
    ensure_structure
    if command -v conda >/dev/null 2>&1 && [[ -f "$PROJECT_ROOT/environment.yml" ]]; then
        log "Updating conda environment from environment.yml"
        conda env update -f "$PROJECT_ROOT/environment.yml"
    elif [[ -f "$PROJECT_ROOT/requirements.txt" ]]; then
        log "Installing pip requirements"
        python -m pip install -r "$PROJECT_ROOT/requirements.txt"
    else
        log "No dependency manifest found. Ensure the 'nutribench' env is active."
    fi
    log "Ensuring expected directories exist."
}

prepare_data() {
    ensure_structure
    python - "$DATA_DIR" <<'PY'
import pathlib
import sys

import pandas as pd

data_dir = pathlib.Path(sys.argv[1])
raw_candidates = [
    data_dir / "nutribench.csv",
    data_dir / "raw.csv",
    data_dir / "raw" / "nutribench.csv",
]

raw_file = next((path for path in raw_candidates if path.exists()), None)
splits = {name: data_dir / f"{name}.csv" for name in ("train", "val", "test")}

if raw_file is None:
    missing = [path for path in splits.values() if not path.exists()]
    if missing:
        missing_str = ", ".join(str(p) for p in missing)
        sys.exit(f"No raw dataset found. Please place nutribench.csv in {data_dir}. Missing: {missing_str}")
    print("Existing splits detected. Nothing to do.")
    sys.exit(0)

df = pd.read_csv(raw_file)
if "query" not in df.columns or "label" not in df.columns:
    sys.exit(f"Raw dataset {raw_file} must contain 'query' and 'label' columns.")

df = df.dropna(subset=["query", "label"]).sample(frac=1, random_state=42).reset_index(drop=True)
n = len(df)
train_end = int(n * 0.8)
val_end = int(n * 0.9)
splits_data = {
    "train": df.iloc[:train_end],
    "val": df.iloc[train_end:val_end],
    "test": df.iloc[val_end:],
}

for name, split_df in splits_data.items():
    out_path = splits[name]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    split_df.to_csv(out_path, index=False)
    print(f"[+] Saved {name} split -> {out_path} ({len(split_df)} rows)")
PY
}

run_benchmarks() {
    ensure_structure
    require_data_splits
    log "Running mean baseline..."
    python "$SCRIPT_DIR/Baseline_average_Guess_Classical_Method.py"
    log "Running L1 (Lasso) regression baseline..."
    python "$SCRIPT_DIR/Linear_Regression_Classical_Method_L1.py"
    log "Running L2 (Ridge) regression baseline..."
    python "$SCRIPT_DIR/Linear_Regression_Classical_Method_L2.py"
}

train_model() {
    ensure_structure
    require_data_splits
    local target="${1:-transformer}"
    if [[ $# -gt 0 ]]; then
        shift
    fi
    case "$target" in
        transformer)
            log "Launching Optuna-powered transformer fine-tuning."
            python "$SCRIPT_DIR/train_transformer.py" "$@"
            ;;
        mean|baseline)
            python "$SCRIPT_DIR/Baseline_average_Guess_Classical_Method.py" "$@"
            ;;
        l1|lasso)
            python "$SCRIPT_DIR/Linear_Regression_Classical_Method_L1.py" "$@"
            ;;
        l2|ridge)
            python "$SCRIPT_DIR/Linear_Regression_Classical_Method_L2.py" "$@"
            ;;
        *)
            log "Unknown train target '$target'. Expected transformer|mean|l1|l2."
            exit 1
            ;;
    esac
}

evaluate_model() {
    ensure_structure
    require_data_splits
    local checkpoint=""
    local split="test"
    local custom_output=""
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --checkpoint)
                checkpoint="$2"
                shift 2
                ;;
            --split)
                split="$2"
                shift 2
                ;;
            --output)
                custom_output="$2"
                shift 2
                ;;
            --save-preds)
                log "Note: --save-preds is no longer required; predictions are always saved."
                shift
                ;;
            *)
                log "Unknown evaluate option '$1'"
                usage
                exit 1
                ;;
        esac
    done

    if [[ -z "$checkpoint" ]]; then
        checkpoint="$(find_latest_checkpoint || true)"
        if [[ -z "$checkpoint" ]]; then
            log "No checkpoints were found. Provide one via --checkpoint."
            exit 1
        fi
        log "Auto-selected checkpoint: $checkpoint"
    fi

    local output_path=""
    if [[ -n "$custom_output" ]]; then
        output_path="$custom_output"
    else
        output_path="$RESULTS_DIR/${split}_predictions.csv"
    fi

    local args=(evaluate --checkpoint "$checkpoint" --split "$split" --output "$output_path")

    log "Saving predictions to $output_path"
    python "$SCRIPT_DIR/evaluate_transformer.py" "${args[@]}"
}

predict_model() {
    ensure_structure
    local checkpoint=""
    local text=""
    local batch_size=""

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --checkpoint)
                checkpoint="$2"
                shift 2
                ;;
            --text)
                text="$2"
                shift 2
                ;;
            --batch-size)
                batch_size="$2"
                shift 2
                ;;
            *)
                log "Unknown predict option '$1'"
                usage
                exit 1
                ;;
        esac
    done

    if [[ -z "$checkpoint" ]]; then
        checkpoint="$(find_latest_checkpoint || true)"
        if [[ -z "$checkpoint" ]]; then
            log "No checkpoints were found. Provide one via --checkpoint."
            exit 1
        fi
        log "Auto-selected checkpoint: $checkpoint"
    fi

    local args=(predict --checkpoint "$checkpoint")
    if [[ -n "$text" ]]; then
        args+=(--text "$text")
    fi
    if [[ -n "$batch_size" ]]; then
        args+=(--batch-size "$batch_size")
    fi
    python "$SCRIPT_DIR/evaluate_transformer.py" "${args[@]}"
}

visualize_data() {
    ensure_structure
    require_data_splits
    log "Generating plots (saved by matplotlib if configured)."
    MPLBACKEND=Agg python "$SCRIPT_DIR/plot_data.py"
}

if [[ $# -lt 1 ]]; then
    usage
    exit 1
fi

MODE="$1"
shift

case "$MODE" in
    setup) setup_env "$@" ;;
    data) prepare_data "$@" ;;
    benchmark) run_benchmarks "$@" ;;
    train) train_model "$@" ;;
    evaluate) evaluate_model "$@" ;;
    predict) predict_model "$@" ;;
    viz|visualize) visualize_data "$@" ;;
    help|-h|--help) usage ;;
    *)
        log "Unknown mode '$MODE'"
        usage
        exit 1
        ;;
esac
