from io import StringIO
import zipfile
import numpy as np
import pandas as pd
import pytest
from generate_data import make_dataset
from src.data import clean_data


def write_csv(tmp_path, frame):
    p = tmp_path / "input.csv"
    frame.to_csv(p, index=False)
    return p


def test_generator_repeatable():
    pd.testing.assert_frame_equal(make_dataset(30, 42), make_dataset(30, 42))
    assert not make_dataset(30, 42).equals(make_dataset(30, 43))


def test_generator_shape_and_units():
    frame = make_dataset(30)
    assert len(frame) == 720
    assert frame.Energy_kW.dropna().min() >= 0
    assert frame.Datetime.is_monotonic_increasing
    assert frame.Energy_kW.isna().sum() == int(720*.005)


def test_generator_requires_history():
    with pytest.raises(ValueError):
        make_dataset(7)


def test_causal_bounded_fill_and_mask(tmp_path):
    raw = pd.DataFrame({"Datetime": pd.date_range("2024-01-01", periods=8, freq="h"),
                        "Energy_kW": [10, np.nan, np.nan, np.nan, np.nan, 20, -1, 30]})
    clean, report = clean_data(write_csv(tmp_path, raw))
    assert clean.Energy_kW.iloc[1:4].eq(10).all()
    assert np.isnan(clean.Energy_kW.iloc[4])
    assert clean.Energy_kW.iloc[6] == 20
    assert not clean.target_observed.iloc[1:5].any()
    assert not clean.target_observed.iloc[6]
    assert report["invalid_power_readings"] == 1
    assert report["remaining_missing_target_hours"] == 1
    assert np.allclose(clean.Energy_kWh, clean.Energy_kW, equal_nan=True)


def test_duplicate_and_subhourly_power_is_mean_not_sum(tmp_path):
    raw = pd.DataFrame({"Datetime": ["2024-01-01 01:00", "2024-01-01 00:00", "2024-01-01 00:00", "2024-01-01 00:30"],
                        "Energy_kW": [8, 2, 4, 6]})
    clean, report = clean_data(write_csv(tmp_path, raw))
    assert clean.Energy_kW.iloc[0] == 4
    assert clean.index.is_monotonic_increasing and clean.index.is_unique
    assert report["duplicate_timestamps_aggregated"] == 1


def test_future_change_does_not_change_earlier_cleaning(tmp_path):
    frame = pd.DataFrame({"Datetime": pd.date_range("2024-01-01", periods=6, freq="h"),
                          "Energy_kW": [4, np.nan, np.nan, 9, 10, 11]})
    first, _ = clean_data(write_csv(tmp_path, frame))
    frame.loc[3:, "Energy_kW"] = 1000
    second, _ = clean_data(write_csv(tmp_path, frame))
    pd.testing.assert_series_equal(first.Energy_kW.iloc[:3], second.Energy_kW.iloc[:3])


@pytest.mark.parametrize("frame", [pd.DataFrame({"wrong": [1]}),
    pd.DataFrame({"Datetime": ["nonsense"], "Energy_kW": [1]}),
    pd.DataFrame({"Datetime": ["2024-01-01"], "Energy_kW": [np.nan]}),
    pd.DataFrame(columns=["Datetime", "Energy_kW"]),
    pd.DataFrame({"Datetime": ["2024-01-01T00:00:00+00:00"], "Energy_kW": [1]})])
def test_bad_csv_fails_clearly(tmp_path, frame):
    with pytest.raises(ValueError):
        clean_data(write_csv(tmp_path, frame))


def test_uci_aggregation_uses_valid_minute_coverage(tmp_path, monkeypatch):
    """Tiny local source archive; no web request and no 126 MB extraction."""
    import src.data as module
    monkeypatch.setattr(module, "path", lambda *p: tmp_path.joinpath(*p))
    cache = tmp_path / ".cache/downloads"
    cache.mkdir(parents=True)
    dt = pd.date_range("2007-01-01", periods=125, freq="min")
    df = pd.DataFrame({"Date": dt.strftime("%d/%m/%Y"), "Time": dt.strftime("%H:%M:%S"),
                       "Global_active_power": [1.0]*60+[2.0]*60+[4.0]*5})
    with zipfile.ZipFile(cache / "uci_household.zip", "w") as z:
        z.writestr("household_power_consumption.txt", df.to_csv(index=False, sep=";"))
    p = module.download_uci()
    out = pd.read_csv(p)
    assert len(out) == 3
    assert out.Energy_kW.iloc[0] == 1.0 and out.Energy_kW.iloc[1] == 2.0
    assert np.isnan(out.Energy_kW.iloc[2])  # a 5-minute partial hour is NOT an hourly truth
    assert not (tmp_path / "household_power_consumption.txt").exists()
    assert p.with_suffix(".metadata.json").exists()
