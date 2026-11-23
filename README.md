# realtime-grafana-dashboard

Live Bitcoin and Ethereum prices in Grafana. A small Python exporter polls the free
[CoinGecko API](https://www.coingecko.com/en/api) every 15 seconds and exposes the
prices as Prometheus metrics. Prometheus scrapes them, and Grafana shows them on a
dashboard that is provisioned automatically, so there is nothing to set up by hand.

```
CoinGecko API ──(every 15s)──> exporter :8000/metrics ──(scrape 15s)──> Prometheus :9090 ──> Grafana :3000
```

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

## Configuration

Set these in your shell or in a `.env` file next to `docker-compose.yml`:

| Variable | Default | Purpose |
|---|---|---|
| `COINGECKO_API_KEY` | _(empty)_ | Optional free CoinGecko **Demo** key. Not required, but raises the rate limit if you hit `429` errors. |
| `GRAFANA_ADMIN_USER` | `admin` | Grafana admin login |
| `GRAFANA_ADMIN_PASSWORD` | `admin` | Grafana admin password |

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
docker-compose.yml              Exporter + Prometheus + Grafana
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
- **Port already in use**: change the left side of the port mapping in
  `docker-compose.yml` (e.g. `"3001:3000"`).
