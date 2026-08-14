"""Bronze Agent's only tool: read raw CSV, write Parquet + metadata.

Preserves original structure (including nulls/dtypes as read) and injects
lineage metadata (load timestamp, source file, row count) without mutating
business data -- no cleansing happens at this layer.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from langchain_core.tools import tool


@tool
def csv_to_bronze_parquet(csv_path: str, output_dir: str) -> str:
    """
    Convert a raw CSV file to Parquet for the Bronze layer. Injects
    lineage metadata columns (_bronze_load_ts, _bronze_source_file) and
    preserves original column values, dtypes, and nulls unmodified.
    Returns the output Parquet file path as a string.
    """
    src = Path(csv_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(src)
    df["_bronze_load_ts"] = datetime.now(timezone.utc).isoformat()
    df["_bronze_source_file"] = src.name

    out_path = out_dir / f"{src.stem}_bronze.parquet"
    df.to_parquet(out_path, index=False)
    return str(out_path)
