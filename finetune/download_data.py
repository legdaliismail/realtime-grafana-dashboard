"""Download hourly Bitcoin prices (BTCUSDT closes) from Binance's free public API."""

import csv
import time

import requests

from common import DATA_FILE, when

URL = "https://data-api.binance.vision/api/v3/klines"  # public market data, no account needed
HOUR_MS = 3_600_000


def main():
    rows, start, session = {}, 0, requests.Session()
    while True:
        for attempt in range(6):
            try:
                resp = session.get(URL, timeout=30, params={
                    "symbol": "BTCUSDT", "interval": "1h", "startTime": start, "limit": 1000})
                if resp.status_code in (418, 429):
                    raise requests.RequestException("rate limited")
                resp.raise_for_status()
                break
            except requests.RequestException as exc:
                if attempt == 5:
                    raise
                print(f"retrying after: {exc}")
                time.sleep(2 ** attempt * 5)
        batch = resp.json()
        now_ms = time.time() * 1000
        for k in batch:
            # k = [open time, open, high, low, close, volume, close time, ...]
            # Skip the hour still in progress. The close belongs to the END of the hour.
            if k[6] < now_ms and float(k[4]) > 0:
                rows[k[0] // HOUR_MS + 1] = k[4]
        if len(batch) < 1000:
            break
        start = batch[-1][0] + HOUR_MS

    DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    with DATA_FILE.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["hour", "close"])
        writer.writerows(sorted(rows.items()))
    hours = sorted(rows)
    missing = hours[-1] - hours[0] + 1 - len(hours)
    print(f"{len(hours):,} hourly prices, {when(hours[0])} to {when(hours[-1])} UTC, {missing} hours missing")


if __name__ == "__main__":
    main()
