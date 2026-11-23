"""Code shared by the live forecaster and the backtest, so both behave identically:
download price history, run Chronos-Bolt, turn the forecast into a signal."""

import os
import time

import numpy as np
import requests
import torch
from chronos import BaseChronosPipeline

HISTORY_URL = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
MODEL_NAME = os.getenv("MODEL_NAME", "amazon/chronos-bolt-small")
API_KEY = os.getenv("COINGECKO_API_KEY", "").strip()
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT_SECONDS", "30"))

HORIZON_HOURS = 6
CONTEXT_HOURS = 720  # how much history the model sees: 30 days of hourly prices
# Chronos-Bolt predicts the 10%..90% quantiles. 10% and 90% give an "80% range":
# the model expects the price to land inside it about 8 times out of 10.
QUANTILES = [0.1, 0.5, 0.9]
SIGNAL_THRESHOLD_PERCENT = float(os.getenv("SIGNAL_THRESHOLD_PERCENT", "0.25"))

HOUR = 3600

session = requests.Session()
session.headers["Accept"] = "application/json"
session.headers["User-Agent"] = "crypto-forecaster/1.0"
if API_KEY:
    session.headers["x-cg-demo-api-key"] = API_KEY


class RateLimited(Exception):
    def __init__(self, retry_after: float):
        super().__init__(f"rate limited, retry after {retry_after:.0f}s")
        self.retry_after = retry_after


class BadData(Exception):
    pass


def fetch_history(days: int) -> list[tuple[float, float]]:
    """Return [(unix_seconds, price)] oldest first. CoinGecko gives hourly points for 2-90 days."""
    if not 2 <= days <= 90:
        raise ValueError("days must be 2-90 (the range where CoinGecko returns hourly data)")
    resp = session.get(HISTORY_URL, params={"vs_currency": "usd", "days": days}, timeout=REQUEST_TIMEOUT)
    if resp.status_code == 429:
        try:
            retry_after = float(resp.headers.get("Retry-After", 60))
        except ValueError:  # the header may also be a date
            retry_after = 60.0
        raise RateLimited(retry_after)
    resp.raise_for_status()
    payload = resp.json()
    prices = payload.get("prices") if isinstance(payload, dict) else None
    if not prices:
        raise BadData("no prices in CoinGecko response")
    points = sorted((ts / 1000, float(price)) for ts, price in prices if price)
    if len(points) < 2:
        raise BadData("too few prices in CoinGecko response")
    return points


def live_context(points: list[tuple[float, float]]) -> np.ndarray:
    """Hourly history ending at the current price."""
    last_ts, last_price = points[-1]
    age = time.time() - last_ts
    if age > HOUR / 2:
        raise BadData(f"newest CoinGecko price is {age / 60:.0f} minutes old")
    # The newest point is "now", not on the hour. If it is too close to the previous
    # hourly point, replace that one so the steps stay roughly one hour apart.
    history = [price for ts, price in points[:-1] if last_ts - ts >= HOUR / 2]
    prices = np.array([*history, last_price], dtype=np.float32)[-CONTEXT_HOURS:]
    if len(prices) < CONTEXT_HOURS // 2:
        raise BadData(f"only {len(prices)} hourly prices available")
    return prices


def load_model():
    return BaseChronosPipeline.from_pretrained(MODEL_NAME, device_map="cpu", dtype=torch.float32)


def predict(model, contexts: np.ndarray) -> np.ndarray:
    """Forecast a batch. contexts: (batch, hours). Returns (batch, 6 hours, [lower, median, upper])."""
    quantiles, _ = model.predict_quantiles(
        torch.tensor(contexts, dtype=torch.float32),
        prediction_length=HORIZON_HOURS,
        quantile_levels=QUANTILES,
    )
    result = np.sort(quantiles.cpu().numpy(), axis=-1)
    if not np.isfinite(result).all():
        raise BadData("model produced a non-finite forecast")
    return result


def change_percent(current, predicted):
    return (predicted - current) / current * 100


def signal_for(change: float) -> int:
    """1 = up, -1 = down, 0 = flat (predicted move is smaller than the threshold)."""
    if change > SIGNAL_THRESHOLD_PERCENT:
        return 1
    if change < -SIGNAL_THRESHOLD_PERCENT:
        return -1
    return 0
