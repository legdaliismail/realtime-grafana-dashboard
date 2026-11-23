"""Fetch live crypto prices from CoinGecko and expose them as Prometheus metrics."""

import logging
import os
import signal
import sys
import threading
import time

import requests
from prometheus_client import Counter, Gauge, Histogram, start_http_server

COINGECKO_URL = "https://api.coingecko.com/api/v3/simple/price"
COINS = {"bitcoin": "BTC", "ethereum": "ETH"}
VS_CURRENCY = "usd"

FETCH_INTERVAL = float(os.getenv("FETCH_INTERVAL_SECONDS", "15"))
METRICS_PORT = int(os.getenv("METRICS_PORT", "8000"))
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT_SECONDS", "10"))
# Optional free "Demo" key from coingecko.com — raises the rate limit. Works without it.
API_KEY = os.getenv("COINGECKO_API_KEY", "").strip()

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger("crypto-exporter")

LABELS = ["coin", "symbol"]
PRICE = Gauge("crypto_price_usd", "Current price in USD", LABELS)
CHANGE_24H = Gauge("crypto_price_change_24h_percent", "Price change over the last 24h, in percent", LABELS)
MARKET_CAP = Gauge("crypto_market_cap_usd", "Market capitalisation in USD", LABELS)
VOLUME_24H = Gauge("crypto_volume_24h_usd", "Trading volume over the last 24h in USD", LABELS)
LAST_UPDATED = Gauge("crypto_last_updated_timestamp_seconds", "CoinGecko's last update time for this price", LABELS)

UP = Gauge("coingecko_up", "1 if the last CoinGecko fetch succeeded, 0 otherwise")
LAST_SUCCESS = Gauge("coingecko_last_success_timestamp_seconds", "Unix time of the last successful fetch")
ERRORS = Counter("coingecko_fetch_errors_total", "Failed CoinGecko fetches", ["reason"])
DURATION = Histogram("coingecko_fetch_duration_seconds", "Time spent calling the CoinGecko API")

session = requests.Session()
session.headers["Accept"] = "application/json"
session.headers["User-Agent"] = "crypto-prometheus-exporter/1.0"
if API_KEY:
    session.headers["x-cg-demo-api-key"] = API_KEY


def fetch_once() -> float:
    """Fetch prices and update gauges. Returns extra seconds to wait (back-off) before next fetch."""
    params = {
        "ids": ",".join(COINS),
        "vs_currencies": VS_CURRENCY,
        "include_market_cap": "true",
        "include_24hr_vol": "true",
        "include_24hr_change": "true",
        "include_last_updated_at": "true",
    }
    try:
        with DURATION.time():
            resp = session.get(COINGECKO_URL, params=params, timeout=REQUEST_TIMEOUT)
        if resp.status_code == 429:
            retry_after = float(resp.headers.get("Retry-After", 60))
            ERRORS.labels("rate_limited").inc()
            UP.set(0)
            log.warning("Rate limited by CoinGecko, backing off %.0fs", retry_after)
            return retry_after
        resp.raise_for_status()
        data = resp.json()
    except requests.Timeout:
        ERRORS.labels("timeout").inc()
        UP.set(0)
        log.warning("CoinGecko request timed out")
        return 0
    except (requests.RequestException, ValueError) as exc:
        ERRORS.labels("request_error").inc()
        UP.set(0)
        log.warning("CoinGecko request failed: %s", exc)
        return 0

    missing = [c for c in COINS if c not in data or VS_CURRENCY not in data[c]]
    if missing:
        ERRORS.labels("bad_payload").inc()
        UP.set(0)
        log.warning("Unexpected payload, missing %s: %s", missing, data)
        return 0

    for coin, symbol in COINS.items():
        d = data[coin]
        PRICE.labels(coin, symbol).set(d[VS_CURRENCY])
        if (v := d.get(f"{VS_CURRENCY}_24h_change")) is not None:
            CHANGE_24H.labels(coin, symbol).set(v)
        if (v := d.get(f"{VS_CURRENCY}_market_cap")) is not None:
            MARKET_CAP.labels(coin, symbol).set(v)
        if (v := d.get(f"{VS_CURRENCY}_24h_vol")) is not None:
            VOLUME_24H.labels(coin, symbol).set(v)
        if (v := d.get("last_updated_at")) is not None:
            LAST_UPDATED.labels(coin, symbol).set(v)

    UP.set(1)
    LAST_SUCCESS.set_to_current_time()
    log.info(
        "BTC=$%s ETH=$%s",
        f"{data['bitcoin'][VS_CURRENCY]:,.2f}",
        f"{data['ethereum'][VS_CURRENCY]:,.2f}",
    )
    return 0


def main() -> None:
    stop = threading.Event()

    def shutdown(signum, _frame):
        log.info("Received signal %s, shutting down", signum)
        stop.set()

    # Needed so `docker compose down` stops us immediately (PID 1 ignores SIGTERM by default).
    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)

    start_http_server(METRICS_PORT)
    log.info("Metrics on :%d/metrics, fetching every %.0fs", METRICS_PORT, FETCH_INTERVAL)

    while not stop.is_set():
        started = time.monotonic()
        backoff = fetch_once()
        elapsed = time.monotonic() - started
        stop.wait(max(FETCH_INTERVAL - elapsed, 0) + backoff)

    sys.exit(0)


if __name__ == "__main__":
    main()
