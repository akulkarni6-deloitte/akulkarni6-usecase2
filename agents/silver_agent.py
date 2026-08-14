from __future__ import annotations

from typing import Optional

from agents.base_agent import BaseSpecialistAgent
from tools.silver_tools import (
    cast_column_dtype,
    deduplicate_rows,
    fill_nulls,
    generate_surrogate_key,
    standardize_date_format,
)
from utils.schemas import STTMDocument


class SilverAgent(BaseSpecialistAgent):
    """Executes the approved Silver STTM: cleansing and standardization."""

    system_prompt = (
        "You are the Silver agent in a medallion data pipeline. Using your "
        "tools (fill_nulls, cast_column_dtype, standardize_date_format, "
        "deduplicate_rows, generate_surrogate_key), execute exactly the "
        "approved Silver STTM rules against the given Bronze Parquet files, "
        "producing cleaned Silver Parquet files. Handle mixed-type columns "
        "and unparseable values gracefully (the tools already do this -- "
        "just call them correctly). Do not perform joins or aggregations."
    )
    max_iterations = 10

    def __init__(self, provider: Optional[str] = None):
        super().__init__(
            tools=[fill_nulls, cast_column_dtype, standardize_date_format,
                   deduplicate_rows, generate_surrogate_key],
            provider=provider,
        )

    def execute(self, sttm: STTMDocument, bronze_parquet_paths: dict[str, str]) -> str:
        result = self.run(
            "Execute this approved Silver STTM against the Bronze Parquet files "
            "listed below. Each cleansing tool operates on a single Parquet "
            "file path and overwrites it in place; call the right tool with "
            "the right file path/column for every rule.\n\n"
            f"Approved STTM rules:\n{sttm.to_execution_json()}\n\n"
            f"Bronze parquet paths by source table: {bronze_parquet_paths}"
        )
        return result["output"]
