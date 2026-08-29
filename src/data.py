"""Raw data handling for Betty.

All raw-data responsibilities live here: loading, schema inspection, column
identification, date/season parsing, result validation, missing-value and
invalid-odds handling, exclusions, and the production of the processed dataset.

Raw files are never modified.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
PROCESSED_PATH = PROCESSED_DIR / "laliga_processed.parquet"

# Seasons excluded from the primary analysis, kept for sensitivity work.
#   2018-19 : no Bet365 *closing* odds columns (B365CH/CD/CA) available.
#   2019-20 : disrupted by COVID-19.
EXCLUDED_PRIMARY = {"18/19", "19/20"}

# Column names for the market of interest (1X2).
COL_DATE = "Date"
COL_HOME = "HomeTeam"
COL_AWAY = "AwayTeam"
COL_RESULT = "FTR"  # H / D / A

# Bet365 odds. Fields named B365H/D/A are the Bet365 odds present in every
# file. Fields B365CH/CD/CA are the Bet365 *closing* odds, only present in the
# more modern layouts (2019-20 onwards).
B365_ODDS = {"H": "B365H", "D": "B365D", "A": "B365A"}
B365_CLOSING_ODDS = {"H": "B365CH", "D": "B365CD", "A": "B365CA"}

VALID_RESULTS = {"H", "D", "A"}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def raw_files() -> list[Path]:
    """Return the raw CSV files, sorted by season."""
    return sorted(RAW_DIR.glob("*.csv"))


def load_raw() -> pd.DataFrame:
    """Load every raw CSV into one long DataFrame with a `season` column.

    The season is taken from the file name, which is the authoritative label
    (e.g. "La Liga Primera Division 2018-19.csv" -> "18/19").
    """
    frames = []
    with warnings.catch_warnings():
        # Adding a column to a freshly-read frame triggers a benign pandas
        # PerformanceWarning about fragmentation; suppress it (tiny data).
        warnings.simplefilter("ignore", pd.errors.PerformanceWarning)
        for path in raw_files():
            season = season_from_filename(path.name)
            df = pd.read_csv(path).assign(season=season)
            frames.append(df)
        return pd.concat(frames, ignore_index=True)


def season_from_filename(name: str) -> str:
    """Extract a short season label from a raw file name."""
    name = name.replace(".csv", "").strip()
    ending = name.split()[-1]  # e.g. "2018-19"
    y0, y1 = ending.split("/")[-1].split("-")
    return f"{y0[-2:]}/{y1[-2:]}"


# ---------------------------------------------------------------------------
# Inspection
# ---------------------------------------------------------------------------

def inspect() -> dict:
    """Produce a concise summary of the raw dataset."""
    df = load_raw()
    summary = {
        "files": [p.name for p in raw_files()],
        "raw_columns": _merged_columns(),
        "n_rows": len(df),
        "season_counts": df["season"].value_counts().sort_index().to_dict(),
        "has_b365_closing": {s: _season_has_closing(s) for s in df["season"].unique()},
    }
    return summary


def _merged_columns() -> list:
    """Union of all column names across files (some files have extra columns)."""
    cols: list[str] = []
    for path in raw_files():
        with pd.io.common.get_handle(path, "r") as f:
            header = f.handle.readline()
        for c in header.strip().split(","):
            c = c.strip().lstrip("\ufeff")
            if c and c not in cols:
                cols.append(c)
    return cols


def _season_has_closing(season: str) -> bool:
    path = _path_for_season(season)
    if path is None:
        return False
    with pd.io.common.get_handle(path, "r") as f:
        header = f.handle.readline()
    return "B365CH" in header


def _path_for_season(season: str) -> Path | None:
    for p in raw_files():
        if season_from_filename(p.name) == season:
            return p
    return None


# ---------------------------------------------------------------------------
# Column identification
# ---------------------------------------------------------------------------

def available_odds_columns(df: pd.DataFrame) -> dict:
    """Report which Bet365 odds columns are present as actual columns."""
    present = {k: set(df.columns) & set(v.values()) for k, v in
               [("regular", B365_ODDS), ("closing", B365_CLOSING_ODDS)]}
    return present


# ---------------------------------------------------------------------------
# Processing
# ---------------------------------------------------------------------------

def process() -> pd.DataFrame:
    """Build and save the processed dataset.

    Primary dataset = only seasons with true Bet365 closing odds
    (2020-21 .. 2025-26); 2018-19 and 2019-20 are excluded as they lack closing
    odds / were COVID-disrupted.
    """
    df = load_raw()
    df = _clean(df)
    df = _infer_season_year(df)
    primary = df[~df["season"].isin(EXCLUDED_PRIMARY)].copy()
    validate_processed(primary)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    primary.to_parquet(PROCESSED_PATH, index=False)
    return primary


def load_processed() -> pd.DataFrame:
    """Load the processed dataset, or process it if missing."""
    if not PROCESSED_PATH.exists():
        return process()
    return pd.read_parquet(PROCESSED_PATH)


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    """Parse dates, standardise text, validate/normalise the 1X2 result."""
    df = df.copy()
    df[COL_DATE] = pd.to_datetime(df[COL_DATE], dayfirst=True, format="mixed",
                                  errors="coerce")
    for c in (COL_HOME, COL_AWAY):
        df[c] = df[c].astype("string").str.strip()
    df[COL_RESULT] = df[COL_RESULT].astype("string").str.strip().str.upper()
    return df


def _infer_season_year(df: pd.DataFrame) -> pd.DataFrame:
    """Add a proper calendar season label and start year based on the date."""
    # File labels already give a short season tag; also record the start year.
    def start_year(season: str) -> int:
        return 2000 + int(season.split("/")[0])

    df["season_start"] = df["season"].map(start_year)
    df["season_label"] = df["season"]
    return df


def validate_processed(df: pd.DataFrame) -> None:
    """Basic sanity checks on the processed primary dataset."""
    if df.empty:
        raise ValueError("Processed dataset is empty.")
    invalid_result = df[~df[COL_RESULT].isin(VALID_RESULTS)]
    if not invalid_result.empty:
        raise ValueError(f"{len(invalid_result)} rows have invalid FTR values.")
    for o in ("H", "D", "A"):
        col = B365_CLOSING_ODDS[o]
        n_bad = pd.to_numeric(df[col], errors="coerce").isna().sum()
        if n_bad:
            raise ValueError(f"{col} has {n_bad} non-numeric / missing values.")


def process_all_seasons() -> pd.DataFrame:
    """Return *all* seasons (including excluded ones) with cleaned fields.

    Used only for sensitivity analysis. Excluded seasons keep whatever Bet365
    odds they have (regular odds for 2018-19; closing odds for 2019-20), flagged
    by a boolean column `has_closing_odds`.
    """
    df = load_raw()
    df = _clean(df)
    df = _infer_season_year(df)
    df["has_closing_odds"] = df["season"].isin(
        [s for s in df["season"].unique() if _season_has_closing(s)]
    )
    return df
