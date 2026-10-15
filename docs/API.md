# Forecast API and dashboard

## Start

```bash
python main.py --dataset synthetic
python main.py --dataset uci
python app.py
```

Base URL on your own machine: `http://127.0.0.1:5000`. The server binds to `0.0.0.0`
so a controlled preview/reverse proxy can reach it. The browser UI uses relative
URLs; it never calls a separate backend at the browser's localhost.

This is a historical-data demo API, **not** a live meter feed. Model bundles and
processed data in the download allow immediate startup. A fresh Git clone needs
a run first because joblib/processed files are gitignored.

The Flask development server has no auth/TLS/rate limiting and is not production
infrastructure. Do not expose private data. See [SECURITY.md](../SECURITY.md).

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/` | Responsive dashboard |
| GET | `/health` | Server health and trained run IDs |
| GET | `/api/runs` | Available sources, model labels and historical cutoffs |
| GET | `/api/summary?dataset=synthetic` | Quality, metrics, stored forecast/scenario metadata |
| GET | `/api/history?dataset=synthetic&hours=168` | Last 24–8,760 processed hourly observations |
| GET | `/api/forecast?dataset=synthetic&hours=24` | Recursive 1–168-hour forecast |
| POST | `/predict` | Same forecast using a JSON body |
| GET | `/api/evaluation?dataset=synthetic` | Candidate comparison and example held-out curves |
| POST | `/api/simulate` | Hypothetical first-24-hour load shifting |
| GET | `/download/<run>/<allowlisted-file>` | Download actual CSV/JSON artifacts |
| GET | `/images/<run>/<allowlisted-image>` | Actual generated analytical PNG |

`dataset` refers to a trained **run ID**: default `synthetic`, measured `uci`, or a
valid custom `--run-id`. Only IDs discovered from completed local runs are accepted.
No user-supplied file path, model upload or arbitrary pickle is accepted.

## Forecast request

### macOS / Linux curl

```bash
curl -s -X POST http://127.0.0.1:5000/predict \
  -H 'Content-Type: application/json' \
  -d '{"dataset":"synthetic","hours":24}'
```

### Windows PowerShell

```powershell
$payload = @{dataset='synthetic'; hours=24} | ConvertTo-Json
$result = Invoke-RestMethod -Uri 'http://127.0.0.1:5000/predict' `
  -Method Post -ContentType 'application/json' -Body $payload
$result.cutoff
$result.predictions | Select-Object -First 5
```

### Python requests

```python
import requests
response = requests.post(
    'http://127.0.0.1:5000/predict',
    json={'dataset': 'uci', 'hours': 24},
    timeout=30,
)
response.raise_for_status()
forecast = response.json()
print(forecast['cutoff'], forecast['model'])
print(forecast['predictions'][0])
```

### Browser GET

Open `http://127.0.0.1:5000/api/forecast?dataset=synthetic&hours=24`.
An address bar sends GET; simply opening POST `/predict` does not send JSON and
will correctly return method-not-allowed. Use curl, requests or Postman for POST.

## Response meaning

Abridged actual synthetic response (remaining prediction rows omitted):

```json
{
  "dataset": "synthetic",
  "model": "mlp",
  "cutoff": "2024-12-30 23:00:00",
  "hours": 24,
  "total_kwh": 2543.52886829057,
  "peak_kw": 131.9643768324111,
  "peak_time": "2024-12-31 19:00:00",
  "source_type": "synthetic",
  "predictions": [
    {
      "Datetime": "2024-12-31 00:00:00",
      "horizon": 1,
      "predicted_kw": 104.116,
      "predicted_kwh": 104.116,
      "lower_kw": 90.454,
      "upper_kw": 117.779
    }
  ]
}
```

The first row is rounded here for readability; actual API values have full floating
precision. Full actual snapshots are in `outputs/api_examples.json`.

- `cutoff`: last historical observation used; **not** today's clock.
- `Datetime`: next hourly interval label in the original dataset clock.
- `horizon`: number of hours after cutoff.
- `predicted_kw`: nonnegative predicted average power during that hour.
- `predicted_kwh`: power × 1 hour, numerically equal for this hourly grid.
- `lower_kw`/`upper_kw`: descriptive empirical validation error band for h≤24;
  `null` beyond 24 because longer horizons are not calibrated here.
- `total_kwh`/`peak_kw`: aggregate over the requested horizon, not always 24 h.

The model needs the previous 168 hourly values saved in the bundle. Arbitrary
`{"hour":14,"day":2}` is intentionally rejected: it is insufficient context for
the model trained here. It also hides which date/cutoff the forecast belongs to.

## Virtual scenario request

```bash
curl -s -X POST http://127.0.0.1:5000/api/simulate \
  -H 'Content-Type: application/json' \
  -d '{"dataset":"synthetic","settings":{"flexible_fraction":0.20,"offpeak_rate":5,"standard_rate":8,"peak_rate":12}}'
```

Optional settings:

| Key | Default | Validation |
|---|---:|---|
| `flexible_fraction` | 0.10 | 0–0.50 |
| `offpeak_rate` | 5 | 0≤offpeak≤standard≤peak≤1000 |
| `standard_rate` | 8 | Same ordered-rate constraint |
| `peak_rate` | 12 | Same ordered-rate constraint |
| `carbon_factor_kg_per_kwh` | 0.70 | 0–5; illustrative, not a sourced regional factor |

Numbers must be finite numeric values, not booleans. Rates are INR/kWh and
illustrative, including for the French public data; do not imply a real tariff.
Peak hours: 17–21 inclusive. Off-peak hours: 0–5 inclusive. Other hours standard.

The scenario uses the first **24 forecast hours**, irrespective of the explorer's
selected horizon. It may pre-schedule or defer flexible work anywhere in that
window. It never exceeds the original predicted global peak. Returned fields
include baseline/shifted energy, peak, cost, moved kWh, hypothetical saving and all
assumptions. Energy is conserved; constant carbon intensity gives zero modeled
emissions reduction. This is a heuristic, not an equipment-control optimizer.

## Validation / errors

```bash
# Unsupported horizon → 400
curl -s 'http://127.0.0.1:5000/api/forecast?hours=999'
# Insufficient old-example input → 400
curl -s -X POST http://127.0.0.1:5000/predict \
  -H 'Content-Type: application/json' -d '{"hour":14,"day":2}'
```

Typical response:

```json
{"error":"invalid_request","message":"hours must be an integer in 1..168."}
```

- 400: invalid object/schema, unknown trained dataset, unsupported parameters,
  malformed JSON or scenario settings.
- 405: unsupported HTTP method.
- 409: a required saved artifact is missing.
- 413: body exceeds 16 KiB.

Valid responses never emit nonstandard JSON NaN/Infinity. Missing numeric values
are `null`. Model/forecast caches are keyed by run and bundle modification time,
so a retrained bundle can invalidate the serving cache.

## Changing port

macOS/Linux: `PORT=5001 python app.py`.
PowerShell: `$env:PORT='5001'; python app.py`.
Open the matching port. A proxied preview URL is provided by the hosting platform;
frontend requests remain relative to that origin.

## Offline HTML

```bash
python scripts/export_dashboard.py
```

Open `outputs/dashboard.html`. It embeds actual precomputed forecasts/results and
supports both datasets, horizon controls, local scenario recomputation and CSV/JSON
exports in a normal browser. It is labeled **interactive snapshot**, not live API.
Some sandboxed file viewers may block browser-triggered downloads; download the
HTML and open it normally, or use the ZIP's original artifacts.
