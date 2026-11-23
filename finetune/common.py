"""Shared paths and helpers for the fine-tuning scripts."""

import csv
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
# Reuse the live forecaster's code, so tests measure what actually runs.
sys.path.insert(0, str(ROOT / "forecaster"))
import forecasting as fc  # noqa: E402

BASE_MODEL = "amazon/chronos-bolt-small"
DATA_FILE = ROOT / "finetune" / "data" / "btc_hourly.csv"
MODEL_DIR = ROOT / "models" / "btc-chronos-bolt-small"
TRAINING_FILE = MODEL_DIR / "training.json"
EVALUATION_FILE = MODEL_DIR / "evaluation.json"

WINDOW = fc.CONTEXT_HOURS + fc.HORIZON_HOURS


def device() -> str:
    """GPU if it really works, otherwise CPU."""
    if torch.cuda.is_available():
        try:
            torch.ones(8, device="cuda").sum().item()
            return "cuda"
        except RuntimeError as exc:
            print(f"GPU found but not usable ({exc}), falling back to CPU")
    return "cpu"


def when(hour) -> str:
    return datetime.fromtimestamp(int(hour) * fc.HOUR, timezone.utc).strftime("%Y-%m-%d %H:%M")


def load_prices():
    """(hours, prices). An hour is an integer: unix time / 3600. Prices are the close at that time."""
    with DATA_FILE.open(newline="") as f:
        rows = list(csv.reader(f))[1:]
    return np.array([int(r[0]) for r in rows]), np.array([float(r[1]) for r in rows], dtype=np.float32)


def gap_free_starts(hours, size=WINDOW):
    """Start index of every window of this size with no missing hours."""
    i = np.arange(len(hours) - size + 1)
    return i[hours[i + size - 1] - hours[i] == size - 1]


def windows(prices, starts):
    """(past the model sees, 6 hours to predict)"""
    chunk = prices[np.asarray(starts)[:, None] + np.arange(WINDOW)]
    return chunk[:, :fc.CONTEXT_HOURS], chunk[:, fc.CONTEXT_HOURS:]


def forecast_median(model, contexts, batch=64):
    parts = [fc.predict(model, contexts[i:i + batch])[:, :, 1] for i in range(0, len(contexts), batch)]
    return np.concatenate(parts)
