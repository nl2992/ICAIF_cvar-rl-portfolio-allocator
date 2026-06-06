from __future__ import annotations

import io
import time
import urllib.request

import pandas as pd

_FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}&cosd={start}&coed={end}"
_HEADERS = {"User-Agent": "Mozilla/5.0 (research; crlpa macro loader)"}

# Default macro state: term spread, high-yield credit spread (OAS), equity vol.
DEFAULT_MACRO = {
    "T10Y2Y": "term_spread",
    "BAMLH0A0HYM2": "credit_spread",
    "VIXCLS": "vix",
}


def fetch_fred_series(series_id: str, start: str, end: str, retries: int = 4) -> pd.Series:
    """Fetch a single FRED series as a date-indexed float Series (no API key)."""
    url = _FRED_CSV.format(series=series_id, start=start, end=end)
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=_HEADERS)
            with urllib.request.urlopen(req, timeout=30) as resp:
                raw = resp.read().decode("utf-8")
            df = pd.read_csv(io.StringIO(raw))
            df.columns = ["date", "value"]
            df["date"] = pd.to_datetime(df["date"])
            df["value"] = pd.to_numeric(df["value"], errors="coerce")  # '.' -> NaN
            return df.set_index("date")["value"].rename(series_id).dropna()
        except Exception as exc:  # network / parse errors -> retry
            last_err = exc
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"failed to fetch FRED {series_id}: {last_err}")


def load_macro(
    series: dict[str, str] | None = None,
    start: str = "2007-01-01",
    end: str = "2024-12-31",
) -> pd.DataFrame:
    """Load several FRED series into a daily, date-indexed, renamed DataFrame."""
    series = series or DEFAULT_MACRO
    frames = []
    for series_id, name in series.items():
        s = fetch_fred_series(series_id, start, end).rename(name)
        frames.append(s)
        time.sleep(0.3)
    return pd.concat(frames, axis=1).sort_index()
