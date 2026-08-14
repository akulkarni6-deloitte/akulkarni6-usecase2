"""Tools used exclusively by the Profiler agent.

Single responsibility: inspect raw CSVs, output profile.json. The profiler
never suggests transformations -- that is the STTM Generator's job.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from langchain_core.tools import tool

from utils.schemas import ColumnProfile, DataProfile, TableProfile


_MAX_SAMPLE_VALUES = 1
_MAX_SAMPLE_STR_LEN = 60


def _shrink_sample(v):
    if isinstance(v, str) and len(v) > _MAX_SAMPLE_STR_LEN:
        return v[:_MAX_SAMPLE_STR_LEN] + "..."
    return v


def _profile_dataframe(file_name: str, df: pd.DataFrame) -> TableProfile:
    columns: list[ColumnProfile] = []
    for col in df.columns:
        series = df[col]
        dtype = str(series.dtype)
        null_count = int(series.isna().sum())
        distinct_count = int(series.nunique(dropna=True))
        sample_values = [
            _shrink_sample(v.item() if hasattr(v, "item") else v)
            for v in series.dropna().head(_MAX_SAMPLE_VALUES).tolist()
        ]

        min_value = max_value = mean_value = None
        if pd.api.types.is_numeric_dtype(series):
            non_null = series.dropna()
            if len(non_null):
                min_value = round(float(non_null.min()), 2)
                max_value = round(float(non_null.max()), 2)
                mean_value = round(float(non_null.mean()), 2)

        columns.append(
            ColumnProfile(
                name=col,
                inferred_dtype=dtype,
                null_count=null_count,
                distinct_count=distinct_count,
                min_value=min_value,
                max_value=max_value,
                mean_value=mean_value,
                sample_values=sample_values,
            )
        )
    return TableProfile(file_name=file_name, row_count=len(df), columns=columns)


@tool
def profile_csv_files(csv_paths: list[str]) -> str:
    """
    Read the given raw CSV file paths and produce a JSON data profile
    (column names, inferred dtypes, null counts, distinct counts, and
    min/max/mean for numeric columns). Does not suggest any transformation.
    Returns the profile as a COMPACT JSON string (no pretty-printing) to
    keep this LLM-facing payload as small as possible.
    """
    tables = []
    for path_str in csv_paths:
        path = Path(path_str)
        df = pd.read_csv(path)
        tables.append(_profile_dataframe(path.name, df))
    profile = DataProfile(tables=tables)
    return profile.model_dump_json()  # compact -- no indent


def write_profile_json(profile_json: str, out_path: str | Path) -> Path:
    """Persist the profiler output to profile.json (used by the pipeline runner)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    parsed = json.loads(profile_json)
    out_path.write_text(json.dumps(parsed, indent=2))
    return out_path
