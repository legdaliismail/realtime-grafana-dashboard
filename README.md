# realtime-grafana-dashboard

Live Bitcoin and Ethereum prices in Grafana. A small Python exporter polls the free
[CoinGecko API](https://www.coingecko.com/en/api) every 15 seconds and exposes the
prices as Prometheus metrics. Prometheus scrapes them, and Grafana shows them on a
dashboard that is provisioned automatically, so there is nothing to set up by hand.

```
CoinGecko API ──(every 15s)──> exporter :8000/metrics ──(scrape 15s)──> Prometheus :9090 ──> Grafana :3000

CoinGecko history ──(every 10 min)──> forecaster :8001/metrics ──(scrape 15s)──> Prometheus
```

The **forecaster** predicts the next 6 hours of the Bitcoin price with Amazon's
[Chronos-Bolt](https://huggingface.co/amazon/chronos-bolt-small) model, on the CPU.

## Run it

You need Docker with Compose v2 (Docker Desktop on Windows/macOS works).

```bash
docker compose up --build
```

Then open:

| What | URL | Notes |
|---|---|---|
| Grafana dashboard | http://localhost:3000 | Opens on the **Crypto Prices (live)** dashboard. Viewable without logging in; to edit, sign in as `admin` / `admin`. |
| Prometheus | http://localhost:9090 | Check **Status → Targets**: `crypto-exporter` should be `UP`. |
| Raw metrics | http://localhost:8000/metrics | What the exporter publishes. |
| Forecast metrics | http://localhost:8001/metrics | What the forecaster publishes. |

The first data points show up within about 15–30 seconds. Charts fill in over time.

To stop: `Ctrl+C`, then `docker compose down`. Add `-v` to also delete the stored
Prometheus/Grafana data.

## Metrics

| Metric | Description |
|---|---|
| `crypto_price_usd{coin,symbol}` | Current price in USD |
| `crypto_price_change_24h_percent{coin,symbol}` | 24h price change, % |
| `crypto_market_cap_usd{coin,symbol}` | Market cap in USD |
| `crypto_volume_24h_usd{coin,symbol}` | 24h trading volume in USD |
| `crypto_last_updated_timestamp_seconds{coin,symbol}` | When CoinGecko last updated the price |
| `coingecko_up` | `1` if the last fetch worked, `0` if not |
| `coingecko_fetch_errors_total{reason}` | Failed fetches (`rate_limited`, `timeout`, `request_error`, `bad_payload`) |
| `coingecko_fetch_duration_seconds` | API call latency histogram |

Forecast metrics (port 8001). `horizon` is `1h` to `6h` ahead; `0h` is the actual price the
forecast started from.

| Metric | Description |
|---|---|
| `crypto_forecast_price_usd{coin,symbol,horizon}` | Predicted price (median) |
| `crypto_forecast_price_lower_usd{...}` / `crypto_forecast_price_upper_usd{...}` | Bounds of the 80% range |
| `crypto_forecast_change_percent{coin,symbol}` | Predicted change over 6 hours, % |
| `crypto_forecast_signal{coin,symbol}` | `1` up, `0` flat, `-1` down |
| `crypto_forecast_up` | `1` if the last forecast attempt worked, `0` if not |
| `crypto_forecast_last_success_timestamp_seconds` | When the last forecast was made |
| `crypto_forecast_errors_total{reason}` | Failed attempts |
| `crypto_forecast_duration_seconds` | Model run time histogram |

If no forecast succeeds for 30 minutes, the forecast values are removed so an old forecast
is never shown as a fresh one.

## How accurate is it?

```bash
docker compose run --rm forecaster python backtest.py
```

This replays the last 90 days: it shows the model 30 days of hourly prices, lets it forecast
6 hours, compares with what happened, and repeats every 6 hours. It reports MAE (average
error in dollars), the % error, and a baseline that always says "the price won't change".

Result for 2026-09-24 to 2026-11-23 (239 forecasts, `chronos-bolt-small`):

| Ahead | Model MAE | Model error | "No change" MAE | "No change" error | Price inside range |
|---|---|---|---|---|---|
| 1h | $199 | 0.26% | $176 | 0.23% | 91% |
| 3h | $324 | 0.43% | $308 | 0.40% | 87% |
| 6h | $579 | 0.77% | $554 | 0.74% | 77% |
| Average of 1h-6h | $366 | 0.49% | $348 | 0.46% | 86% |

**The model did not beat "the price won't change" (5.2% higher average error), and the UP/DOWN
signals were about as good as a coin flip** (UP right 33% of 18 times, DOWN right 54% of 41 times).
Short-term Bitcoin prices are close to random. The most useful output is the range, which is meant
to hold the real price 80% of the time. This is a demo, not financial advice.

## Configuration

Set these in your shell or in a `.env` file next to `docker-compose.yml`:

| Variable | Default | Purpose |
|---|---|---|
| `COINGECKO_API_KEY` | _(empty)_ | Optional free CoinGecko **Demo** key. Not required, but raises the rate limit if you hit `429` errors. |
| `GRAFANA_ADMIN_USER` | `admin` | Grafana admin login |
| `GRAFANA_ADMIN_PASSWORD` | `admin` | Grafana admin password |

Forecaster settings are in `docker-compose.yml`: `FORECAST_INTERVAL_SECONDS` (600), `MODEL_NAME`
(`amazon/chronos-bolt-small`) and `SIGNAL_THRESHOLD_PERCENT` (0.25; smaller predicted moves count as flat).

The polling interval is `FETCH_INTERVAL_SECONDS` in `docker-compose.yml`. Don't go much
below 15s without an API key, or CoinGecko will rate-limit you.

## Running the exporter without Docker

```bash
pip install -r requirements.txt
python exporter.py      # metrics on http://localhost:8000/metrics
```

## Project layout

```
exporter.py                     Python exporter (CoinGecko -> Prometheus metrics)
requirements.txt                Python dependencies
Dockerfile                      Image for the exporter
forecaster/forecaster.py        Forecast service (Chronos-Bolt -> Prometheus metrics)
forecaster/forecasting.py       Shared code: price history, model, signal
forecaster/backtest.py          Accuracy test on past data
docker-compose.yml              Exporter + forecaster + Prometheus + Grafana
prometheus/prometheus.yml       Scrape config
grafana/provisioning/           Auto-provisioned data source and dashboard provider
grafana/dashboards/             Dashboard JSON
```

## Troubleshooting

- **Dashboard shows "No data"**: wait 30s, then check http://localhost:9090/targets. If the
  exporter is `DOWN`, run `docker compose logs exporter`.
- **`CoinGecko API: DOWN` / rate limited**: the public API without a key is shared and
  sometimes throttled. The exporter backs off automatically; adding `COINGECKO_API_KEY`
  fixes it for good.
- **Forecast panels show "No data"**: the first start downloads the model (about 200 MB). Check
  `docker compose logs forecaster`. The dashed "Predicted 6h earlier" line needs 6 hours of running.
- **Forecast chart or panel missing after updating**: run `docker compose up -d --build` and
  `docker compose restart prometheus`, so Prometheus picks up its new scrape config.
- **Port already in use**: change the left side of the port mapping in
  `docker-compose.yml` (e.g. `"3001:3000"`).
