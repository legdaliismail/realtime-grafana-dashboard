"""Compare the original and the fine-tuned model on data neither was fine-tuned on.

Writes evaluation.json next to the fine-tuned model. The live forecaster reads it and uses the
fine-tuned model only if it won. Rules (fixed before seeing any result):
  1. Binance test period (last 12 months): lower MAE, lower % error, and the gap is larger than luck
     (95% bootstrap interval of the % error gap is above zero).
  2. CoinGecko, last 90 days (the live data source): lower MAE and lower % error too.
"""

import json
import time

import numpy as np
import torch
from chronos import BaseChronosPipeline

from common import (BASE_MODEL, EVALUATION_FILE, MODEL_DIR, TRAINING_FILE, device, fc,
                    forecast_median, gap_free_starts, load_prices, when, windows)
import backtest  # forecaster/backtest.py: CoinGecko download and window code

STRIDE = 1  # forecast from every hour, so the result does not depend on the starting hour
BLOCK = 7 * 24  # bootstrap resamples whole weeks, because nearby forecasts overlap
RESAMPLES = 5000


def load(name, dev):
    return BaseChronosPipeline.from_pretrained(str(name), device_map=dev, dtype=torch.float32)


def significant_gap(original_pct, tuned_pct):
    """95% interval of (original - fine-tuned) % error, per forecast. Positive = fine-tuned is better."""
    gap = (original_pct - tuned_pct).mean(axis=1)
    blocks = [gap[i:i + BLOCK] for i in range(0, len(gap), BLOCK)]
    rng = np.random.default_rng(0)
    means = [np.concatenate([blocks[j] for j in rng.integers(0, len(blocks), len(blocks))]).mean()
             for _ in range(RESAMPLES)]
    return float(gap.mean()), [float(v) for v in np.percentile(means, [2.5, 97.5])]


def compare(title, contexts, actuals, models):
    """Prints a table. Returns {model: {mae, error_percent}} plus the per-forecast % errors."""
    predictions = {name: forecast_median(m, contexts) for name, m in models.items()}
    predictions["no change (baseline)"] = np.repeat(contexts[:, -1:], fc.HORIZON_HOURS, axis=1)
    print(f"\n{title}: {len(contexts)} forecasts, average of 1h-6h ahead")
    print(f"  {'model':<22} {'MAE':>8} {'error %':>9}   {'6h MAE':>8} {'6h error %':>11}")
    scores, pct = {}, {}
    for name, pred in predictions.items():
        err = np.abs(pred - actuals)
        pct[name] = err / actuals * 100
        scores[name] = {"mae": float(err.mean()), "error_percent": float(pct[name].mean())}
        print(f"  {name:<22} {'$' + format(err.mean(), ',.1f'):>8} {pct[name].mean():>8.4f}%   "
              f"{'$' + format(err[:, -1].mean(), ',.1f'):>8} {pct[name][:, -1].mean():>10.4f}%")
    return scores, pct


def better(scores):
    o, t = scores["original"], scores["fine-tuned"]
    return t["mae"] < o["mae"] and t["error_percent"] < o["error_percent"]


def main():
    dev = device()
    info = json.loads(TRAINING_FILE.read_text())
    test_start = info["test_start"]
    models = {"original": load(BASE_MODEL, dev), "fine-tuned": load(MODEL_DIR, dev)}

    hours, prices = load_prices()
    starts = gap_free_starts(hours)
    starts = starts[hours[starts + fc.CONTEXT_HOURS] >= test_start][::STRIDE]
    x, y = windows(prices, starts)
    print(f"Test data: {when(test_start)} to {when(hours[-1])} UTC (never used in training)")
    binance, pct = compare("Binance test period", x, y, models)
    gap, interval = significant_gap(pct["original"], pct["fine-tuned"])
    print(f"  fine-tuned is better by {gap:.4f} points of % error "
          f"(95% interval {interval[0]:+.4f} to {interval[1]:+.4f})")

    # Second check on CoinGecko prices, what the live forecaster really receives.
    cg_hours, cg_prices = backtest.hourly_series(backtest.fetch_with_retry(90))
    cx, cy, cs, _ = backtest.build_windows(cg_hours, cg_prices, STRIDE)
    if len(cs) and cs[0] + 1 < test_start:
        raise SystemExit("CoinGecko data overlaps the training period; train again on newer data first")
    coingecko, _ = compare("CoinGecko, last 90 days", cx, cy, models)

    checks = {
        "binance_lower_error": better(binance),
        "binance_beyond_luck": interval[0] > 0,
        "coingecko_lower_error": better(coingecko),
    }
    use = all(checks.values())
    print("\nChecks:")
    for name, ok in checks.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    print(f"\nDecision: the forecaster will use the {'FINE-TUNED' if use else 'ORIGINAL'} model.")

    EVALUATION_FILE.write_text(json.dumps({
        "use_finetuned": use, "base_model": BASE_MODEL, "checks": checks,
        "test_start": when(test_start), "test_end": when(hours[-1]), "test_forecasts": len(x),
        "binance": binance, "coingecko": coingecko,
        "gap_error_percent": gap, "gap_95_interval": interval, "evaluated": time.strftime("%Y-%m-%d %H:%M"),
    }, indent=2))


if __name__ == "__main__":
    main()
