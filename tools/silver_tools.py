"""Silver Agent's tools: cleansing and standardization primitives.

Each tool is intentionally narrow so the ReAct agent composes them
according to the approved Silver STTM, and each handles the edge cases
called out in the brief (mixed-type columns, unparseable dates, etc.)
gracefully rather than raising.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import pandas as pd
from langchain_core.tools import tool


def _load(parquet_path: str) -> pd.DataFrame:
    return pd.read_parquet(parquet_path)


def _save(df: pd.DataFrame, parquet_path: str) -> str:
    df.to_parquet(parquet_path, index=False)
    return parquet_path


@tool
def fill_nulls(parquet_path: str, column: str, strategy: Literal["mean", "median", "constant", "mode"],
                constant_value: str | float | None = None) -> str:
    """
    Fill null values in `column` of the Parquet file at `parquet_path`
    using `strategy` ('mean', 'median', 'constant', or 'mode'). For
    'constant', supply `constant_value`. Non-numeric columns fall back to
    'mode' or 'constant' gracefully even if 'mean'/'median' is requested.
    Overwrites the file in place and returns its path.
    """
    df = _load(parquet_path)
    if column not in df.columns:
        return f"ERROR: column '{column}' not found in {parquet_path}"

    series = df[column]
    is_numeric = pd.api.types.is_numeric_dtype(series)

    if strategy in ("mean", "median") and not is_numeric:
        strategy = "mode"  # graceful fallback for mixed/non-numeric columns

    if strategy == "mean":
        fill_value = series.mean()
    elif strategy == "median":
        fill_value = series.median()
    elif strategy == "mode":
        mode_vals = series.mode(dropna=True)
        fill_value = mode_vals.iloc[0] if not mode_vals.empty else constant_value
    else:  # constant
        fill_value = constant_value

    df[column] = series.fillna(fill_value)
    return _save(df, parquet_path)


@tool
def cast_column_dtype(parquet_path: str, column: str,
                       target_dtype: Literal["datetime", "int", "float", "string"]) -> str:
    """
    Cast `column` in the Parquet file to `target_dtype`. Invalid/unparseable
    values become nulls rather than raising (graceful handling of mixed-type
    columns), and a `<column>_cast_errors` count is printed for visibility.
    Overwrites the file in place and returns its path.
    """
    df = _load(parquet_path)
    if column not in df.columns:
        return f"ERROR: column '{column}' not found in {parquet_path}"

    original_non_null = df[column].notna().sum()
    if target_dtype == "datetime":
        df[column] = pd.to_datetime(df[column], errors="coerce")
    elif target_dtype == "int":
        df[column] = pd.to_numeric(df[column], errors="coerce").astype("Int64")
    elif target_dtype == "float":
        df[column] = pd.to_numeric(df[column], errors="coerce")
    else:
        df[column] = df[column].astype("string")

    new_non_null = df[column].notna().sum()
    cast_errors = int(original_non_null - new_non_null)
    _save(df, parquet_path)
    return f"OK: cast {column} to {target_dtype} on {parquet_path} ({cast_errors} values became null)"


@tool
def standardize_date_format(parquet_path: str, column: str) -> str:
    """
    Standardize a date/datetime column to 'YYYY-MM-DD' string format,
    parsing mixed source formats. Unparseable values become null.
    Overwrites the file in place and returns its path.
    """
    df = _load(parquet_path)
    if column not in df.columns:
        return f"ERROR: column '{column}' not found in {parquet_path}"

    parsed = pd.to_datetime(df[column], errors="coerce", format="mixed")
    df[column] = parsed.dt.strftime("%Y-%m-%d")
    return _save(df, parquet_path)


@tool
def deduplicate_rows(parquet_path: str, subset_columns: list[str] | None = None) -> str:
    """
    Remove duplicate records from the Parquet file, optionally restricted
    to `subset_columns` for the duplicate key. Keeps the first occurrence.
    Overwrites the file in place and returns its path with a row-count delta.
    """
    df = _load(parquet_path)
    before = len(df)
    df = df.drop_duplicates(subset=subset_columns, keep="first")
    after = len(df)
    _save(df, parquet_path)
    return f"OK: deduplicated {parquet_path} ({before - after} rows removed, {after} remain)"


@tool
def generate_surrogate_key(parquet_path: str, new_column: str) -> str:
    """
    Add a monotonically increasing surrogate key column `new_column`
    (1-based) to the Parquet file. Overwrites the file in place and
    returns its path.
    """
    df = _load(parquet_path)
    df[new_column] = range(1, len(df) + 1)
    return _save(df, parquet_path)
