# Data card

## 1. Two sources, deliberately separated

| Property | Virtual building | Public measured household |
|---|---|---|
| CLI | `--dataset synthetic` | `--dataset uci` |
| Input | `data/raw/energy_data.csv` | `data/raw/uci_household_hourly.csv` |
| Origin | Seeded generator in this repository | UCI dataset 235, Hebrail & Berard |
| Site | Fictitious mixed-use building | One house in Sceaux, France |
| Original sampling | Hourly | One minute |
| Original records | 17,520 | 2,075,259 |
| Hourly raw rows | 17,520 | 34,589 |
| Cleaned grid rows | 17,520 | 34,587 after trimming partial edges |
| Cleaned start | 2023-01-01 00:00 | 2006-12-16 18:00 |
| Cleaned end | 2024-12-30 23:00 | 2010-11-26 20:00 |
| License | Synthetic data CC0; generator MIT | Data CC BY 4.0 |
| Claims permitted | Simulation results under explicit assumptions | Historical single-household benchmark results |
| Claims not permitted | Corporate/customer data or actual savings | Representative city/grid demand or current behavior |

Dates are dataset dates, not today's date. The 730-day simulation does not include
all of leap-year 2024; it ends on December 30. The dashboard states its cutoff.

## 2. Official public source and attribution

Georges Hebrail and Alice Berard, **Individual Household Electric Power Consumption**,
UCI Machine Learning Repository. DOI https://doi.org/10.24432/C58K54.

Official page: https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption

UCI describes one-minute, minute-averaged active power in kW from a household
between December 2006 and November 2010. Other source variables include voltage,
reactive power and submetering. This project uses **only active power and clock**
for forecasting. It does not attach guessed weather to historical French records.
See [DATA_LICENSE.md](../DATA_LICENSE.md) for redistribution attribution.

The bundled CSV is our hourly adaptation. `--refresh-data` downloads the official
archive over HTTPS into `.cache/downloads`, reads in 250,000-row chunks, combines
hourly sums and counts across chunk boundaries, and writes a small hourly file.
It does not extract the 126.8 MB text into the repository.

### Coverage rule

1. Parse `Date` + `Time` using `%d/%m/%Y %H:%M:%S`.
2. Treat `?`, blank, nonnumeric, negative and infinite power as missing.
3. For each hour, divide sum of valid minute-average kW readings by valid count.
4. Require **at least 54 valid minutes** (90% of an hour); otherwise target = NaN.
5. Save `Observed_minutes` and `Source_minutes` so coverage can be audited.
6. Later trim leading/trailing unobserved hours; preserve internal gaps.

An hour based on 54–59 measurements estimates the full-hour mean. This approximation
is acceptable for an educational forecasting benchmark, **not revenue-grade billing**.
There is no fabricated zero-energy hour for a missing block.

## 3. Schema and units

| Field | Meaning | Forecast feature? |
|---|---|---|
| `Datetime` | Start label of the hourly interval in the documented dataset clock | Calendar is derived from it |
| `Energy_kW` | **Hour-average power in kW**, despite the historical column name | Lagged values only |
| `Temperature_C` | Generated hourly temperature; synthetic data only | No; EDA only |
| `Occupancy` | Generated occupancy proxy 0–1; synthetic data only | No; EDA only |
| `IsWeekend` | Generator's weekday/weekend flag | Re-derived from calendar |
| `Observed_minutes` | UCI valid source measurements per hour | Coverage audit only |
| `Source_minutes` | UCI source rows contributing to hour | Coverage audit only |
| `target_observed` | Cleaner flag indicating non-imputed original hourly target | Training/evaluation mask |
| `Energy_kWh` | `Energy_kW × 1 hour` | Reporting only |

**kW is power, kWh is energy.** At a constant mean 2 kW for one hour, energy is
2 kWh. A 24-hour total is the sum of all hourly kWh, not the mean of kW.

Do not feed cumulative meter-register kWh into this schema. First difference
consecutive register readings, handle resets/rollovers, divide by duration to
obtain average kW, and reject invalid durations. Do not sum power readings as if
they were already interval energy. For irregular readings use duration-weighted
aggregation before this pipeline; the generic cleaner assumes regularly sampled
power or correctly prepared hourly inputs.

## 4. Synthetic process

The generator combines:

- base power of 50 kW plus always-on appliance load;
- morning/evening Gaussian-shaped activity peaks;
- a 15% weekend reduction applied to the activity component;
- a daily temperature cosine peaking at 14:00;
- an annual temperature sine and a modest appliance-season factor;
- cooling above 24°C and heating below 16°C;
- business-hours occupancy and random noise;
- 0.5% deliberately missing target readings.

`numpy.random.default_rng(seed)` is created inside the generator, so two calls with
the same seed and length return identical data. This is a **plausible learning
simulation**, not a calibrated building-physics or regional weather model.

Temperature and occupancy help illustrate why demand changes, but their current
future values are not used as model inputs. A real implementation could use
forecast weather and known occupancy schedules issued before the forecast cutoff.

## 5. Causal cleaning and missingness

| Audit | Synthetic | UCI |
|---|---:|---:|
| Missing internal target hours before fill | 87 | 444 |
| Causally filled target hours | 87 | 33 |
| Long-gap hours still missing | 0 | 411 |
| Observed target hours | 17,433 | 34,143 |
| Eligible feature/target rows after warmup/gaps/mask | 17,266 | 32,631 |

Cleaning sorts timestamps, averages duplicate/sub-hourly power values into the
hourly grid, rejects negative/infinite targets, and forward-fills **at most three
hours** using past values. No backward fill or interpolation with future endpoints.

The original observation mask is stored before filling. Imputed values may be
available as past lag inputs, but **are not training labels or test truth**.
Long gaps remain missing. The 168-hour rolling mean requires a complete past week,
so rows affected by long gaps are removed from supervised samples. A forecast
requires 168 contiguous finite recent hourly values; it fails clearly otherwise.

## 6. Time semantics and limitations

- Bundled timestamps are timezone-naive dataset clocks. They are not converted
  into the viewer's location/timezone or treated as live observations.
- UCI's original local calendar labels are preserved. Timezone/DST transitions
  have not been reconstructed; this is a documented benchmark limitation.
- Timezone-aware custom inputs are rejected. Normalize a site clock or UTC and
  document the choice before importing them. Production should use site-aware
  timestamps and explicit DST reconciliation.
- No holidays, tariffs, economic events, equipment faults, occupancy schedules
  or measured weather are available for the real household forecast.
- A single household can be erratic. A building/utility's aggregate series may
  behave differently. Do not transfer accuracy figures between sites.
- No personal identifiers beyond the source's already-public aggregate series
  are introduced. Never add private billing/address data to the repository.

## 7. Your own CSV

Required columns, ISO timestamp format:

```csv
Datetime,Energy_kW
2024-01-01 00:00:00,1.25
2024-01-01 01:00:00,1.40
2024-01-01 02:00:00,1.10
```

This is a schema illustration, not enough data to train. Use at least 60 days of
hourly observations; a year or more is preferable for annual seasonality.

```bash
python main.py --dataset csv --csv data/raw/my_building.csv --run-id building
```

Verify units, ownership, time semantics, missingness and the final week's quality
before running it. A private CSV should live outside the Git-tracked data folders.
