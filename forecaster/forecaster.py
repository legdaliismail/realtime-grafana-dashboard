"""Forecast the next 6 hours of Bitcoin prices and expose the result as Prometheus metrics."""

import logging
import os
import signal
import sys
import threading
import time

import requests
from prometheus_client import Counter, Gauge, Histogram, start_http_server

import forecasting as fc

FORECAST_INTERVAL = float(os.getenv("FORECAST_INTERVAL_SECONDS", "600"))
METRICS_PORT = int(os.getenv("METRICS_PORT", "8001"))
RETRY_SECONDS = 60  # after a failure, try again sooner than the normal interval
STALE_AFTER = 3 * FORECAST_INTERVAL
HISTORY_DAYS = 30

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(message)s",
)
# The Hugging Face downloader logs every HTTP request at INFO (logger name varies by version).
for noisy in ("httpx", "httpx2"):
    logging.getLogger(noisy).setLevel(logging.WARNING)
log = logging.getLogger("crypto-forecaster")

LABELS = ["coin", "symbol"]
BTC = ("bitcoin", "BTC")
H_LABELS = [*LABELS, "horizon"]
# horizon="0h" is the actual price the forecast started from; "1h".."6h" are predictions.
PRICE = Gauge("crypto_forecast_price_usd", "Predicted price (median) this far ahead", H_LABELS)
LOWER = Gauge("crypto_forecast_price_lower_usd", "Lower bound of the 80% forecast range", H_LABELS)
UPPER = Gauge("crypto_forecast_price_upper_usd", "Upper bound of the 80% forecast range", H_LABELS)
CHANGE = Gauge("crypto_forecast_change_percent", "Predicted price change over the next 6 hours, in percent", LABELS)
SIGNAL = Gauge("crypto_forecast_signal", "Predicted direction over 6 hours: 1 up, 0 flat, -1 down", LABELS)

UP = Gauge("crypto_forecast_up", "1 if the last forecast attempt succeeded, 0 otherwise")
LAST_SUCCESS = Gauge("crypto_forecast_last_success_timestamp_seconds", "Unix time of the last successful forecast")
ERRORS = Counter("crypto_forecast_errors_total", "Failed forecast attempts", ["reason"])
DURATION = Histogram("crypto_forecast_duration_seconds", "Time spent running the model")

FORECAST_GAUGES = (PRICE, LOWER, UPPER, CHANGE, SIGNAL)


def publish(current: float, forecast) -> None:
    """forecast: (6 hours, [lower, median, upper])"""
    for gauge in (PRICE, LOWER, UPPER):
        gauge.labels(*BTC, "0h").set(current)
    for hour, (lower, median, upper) in enumerate(forecast.tolist(), start=1):
        LOWER.labels(*BTC, f"{hour}h").set(lower)
        PRICE.labels(*BTC, f"{hour}h").set(median)
        UPPER.labels(*BTC, f"{hour}h").set(upper)
    lower, median, upper = forecast[-1].tolist()
    change = fc.change_percent(current, median)
    sig = fc.signal_for(change)
    CHANGE.labels(*BTC).set(change)
    SIGNAL.labels(*BTC).set(sig)
    log.info(
        "BTC now $%s -> 6h $%s (range $%s - $%s), change %+.2f%%, signal %s",
        f"{current:,.0f}", f"{median:,.0f}", f"{lower:,.0f}", f"{upper:,.0f}",
        change, {1: "UP", 0: "FLAT", -1: "DOWN"}[sig],
    )


def forecast_once(model) -> float | None:
    """Make one forecast. Returns None on success, or how many seconds to wait before retrying."""
    try:
        prices = fc.live_context(fc.fetch_history(HISTORY_DAYS))
        with DURATION.time():
            forecast = fc.predict(model, prices[None, :])[0]
        publish(float(prices[-1]), forecast)
    except fc.RateLimited as exc:
        reason, wait = "rate_limited", max(exc.retry_after, RETRY_SECONDS)
        log.warning("CoinGecko: %s", exc)
    except requests.Timeout:
        reason, wait = "timeout", RETRY_SECONDS
        log.warning("CoinGecko request timed out")
    except requests.RequestException as exc:
        reason, wait = "request_error", RETRY_SECONDS
        log.warning("CoinGecko request failed: %s", exc)
    except fc.BadData as exc:
        reason, wait = "bad_data", RETRY_SECONDS
        log.warning("Unusable data: %s", exc)
    except Exception:
        # Keep the service alive: one bad forecast should not take the metrics down.
        reason, wait = "unexpected", RETRY_SECONDS
        log.exception("Forecast failed")
    else:
        UP.set(1)
        LAST_SUCCESS.set_to_current_time()
        return None
    ERRORS.labels(reason).inc()
    UP.set(0)
    return wait


def main() -> None:
    stop = threading.Event()

    def shutdown(signum, _frame):
        log.info("Received signal %s, shutting down", signum)
        stop.set()

    # Needed so `docker compose down` stops us immediately (PID 1 ignores SIGTERM by default).
    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)

    UP.set(0)
    start_http_server(METRICS_PORT)
    log.info("Metrics on :%d/metrics; loading %s (first start downloads it)", METRICS_PORT, fc.MODEL_NAME)
    model = fc.load_model()
    log.info("Model ready, forecasting every %.0fs", FORECAST_INTERVAL)

    last_success = None
    while not stop.is_set():
        started = time.monotonic()
        retry_in = forecast_once(model)
        if retry_in is None:
            last_success = started
            delay = FORECAST_INTERVAL - (time.monotonic() - started)
        else:
            delay = retry_in
            # An old forecast must not look like a fresh one: after a while show nothing instead.
            if last_success is not None and time.monotonic() - last_success > STALE_AFTER:
                for gauge in FORECAST_GAUGES:
                    gauge.clear()
                last_success = None
                log.warning("Forecast is stale, removed it from the metrics")
        stop.wait(max(delay, 0))

    sys.exit(0)


if __name__ == "__main__":
    main()
