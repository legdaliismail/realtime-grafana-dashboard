"""Measure how accurate the forecasts would have been on past data.

Replays history: stand at a past hour, show the model only the prices up to then,
forecast 6 hours, then compare with what really happened. Repeat for many hours.

    docker compose run --rm forecaster python backtest.py
"""

import argparse
import sys
import time
from datetime import datetime, timezone

import numpy as np

import forecasting as fc

BATCH = 32


def fetch_with_retry(days: int, attempts: int = 5):
    for attempt in range(1, attempts + 1):
        try:
            return fc.fetch_history(days)
        except fc.RateLimited as exc:
            if attempt == attempts:
                raise
            wait = max(exc.retry_after, 60)
            print(f"CoinGecko rate limit, waiting {wait:.0f}s ...", flush=True)
            time.sleep(wait)


def hourly_series(points) -> tuple[np.ndarray, np.ndarray]:
    """Snap points to whole hours and drop the trailing 'now' point."""
    by_hour = {}
    for ts, price in points[:-1]:
        by_hour.setdefault(round(ts / fc.HOUR), price)
    hours = np.array(sorted(by_hour))
    return hours, np.array([by_hour[h] for h in hours], dtype=np.float32)


def build_windows(hours, prices, stride):
    """Cut history into (past the model sees, 6 hours it must predict). Skips gaps in the data."""
    contexts, actuals, starts, skipped = [], [], [], 0
    size = fc.CONTEXT_HOURS + fc.HORIZON_HOURS
    for i in range(0, len(prices) - size + 1, stride):
        if hours[i + size - 1] - hours[i] != size - 1:
            skipped += 1
            continue
        split = i + fc.CONTEXT_HOURS
        contexts.append(prices[i:split])
        actuals.append(prices[split:i + size])
        starts.append(hours[split - 1])
    return np.array(contexts), np.array(actuals), starts, skipped


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--days", type=int, default=90, help="days of history to download (max 90)")
    parser.add_argument("--stride", type=int, default=fc.HORIZON_HOURS, help="hours between test forecasts")
    args = parser.parse_args()
    if not 2 <= args.days <= 90:
        parser.error("--days must be between 2 and 90")
    if args.stride < 1:
        parser.error("--stride must be at least 1")

    hours, prices = hourly_series(fetch_with_retry(args.days))
    contexts, actuals, starts, skipped = build_windows(hours, prices, args.stride)
    if not len(contexts):
        print(f"Nothing to test: each forecast needs {fc.CONTEXT_HOURS + fc.HORIZON_HOURS} gap-free hours, "
              f"got {len(prices)} hours ({skipped} windows skipped for gaps).")
        return 1

    print(f"Loading {fc.MODEL_NAME} ...", flush=True)
    model = fc.load_model()
    forecasts = np.concatenate([fc.predict(model, contexts[i:i + BATCH]) for i in range(0, len(contexts), BATCH)])
    lower, median, upper = (forecasts[:, :, q] for q in range(3))
    current = contexts[:, -1:]  # price at the moment of each forecast

    err = np.abs(median - actuals)
    # Baseline: simply say "the price won't change". A useful model must beat this.
    naive_err = np.abs(current - actuals)
    inside = (actuals >= lower) & (actuals <= upper)

    def when(hour):
        return datetime.fromtimestamp(int(hour) * fc.HOUR, timezone.utc).strftime("%Y-%m-%d %H:%M")

    print(f"\n{len(contexts)} forecasts, {when(starts[0])} to {when(starts[-1])} UTC"
          + (f" ({skipped} skipped: gaps in data)" if skipped else ""))
    print(f"\n{'ahead':>7} {'MAE':>8} {'error %':>8} {'no-change MAE':>14} {'no-change %':>12} {'in range':>9}")

    def row(name, e, n, a, ins):
        print(f"{name:>7} {'$' + format(e.mean(), ',.0f'):>8} {(e / a).mean() * 100:>7.2f}% "
              f"{'$' + format(n.mean(), ',.0f'):>14} {(n / a).mean() * 100:>11.2f}% {ins.mean() * 100:>8.0f}%")

    for h in range(fc.HORIZON_HOURS):
        row(f"{h + 1}h", err[:, h], naive_err[:, h], actuals[:, h], inside[:, h])
    row("average", err, naive_err, actuals, inside)

    skill = (1 - err.mean() / naive_err.mean()) * 100
    print(f"\nModel vs saying 'the price won't change': {abs(skill):.1f}% "
          f"{'better' if skill >= 0 else 'worse'} (average MAE)")
    print("The forecast range is meant to contain the real price 80% of the time.")

    predicted = fc.change_percent(current[:, 0], median[:, -1])
    actual = fc.change_percent(current[:, 0], actuals[:, -1])
    signals = np.array([fc.signal_for(c) for c in predicted])
    print(f"\nSignal, checked 6h later (threshold {fc.SIGNAL_THRESHOLD_PERCENT}%):")
    for value, name in ((1, "UP"), (-1, "DOWN")):
        chosen = signals == value
        line = f"  {name:<5} {chosen.sum():>4} times"
        if chosen.any():
            line += f", price moved that way {(np.sign(actual[chosen]) == value).mean() * 100:.0f}% of the time"
        print(line)
    print(f"  FLAT  {(signals == 0).sum():>4} times")
    print(f"  For reference: the price rose in {(actual > 0).mean() * 100:.0f}% of all test periods.")
    print(f"  Typical 6h move: predicted {np.median(np.abs(predicted)):.2f}%, real {np.median(np.abs(actual)):.2f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
