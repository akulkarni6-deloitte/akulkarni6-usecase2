from __future__ import annotations

from typing import Optional

from agents.base_agent import BaseSpecialistAgent
from tools.bronze_tools import csv_to_bronze_parquet
from utils.schemas import STTMDocument


class BronzeAgent(BaseSpecialistAgent):
    """Executes the approved Bronze STTM: CSV -> Parquet + metadata, nothing else."""

    system_prompt = (
        "You are the Bronze agent in a medallion data pipeline. Your only "
        "tool is `csv_to_bronze_parquet`. Given a list of raw CSV paths and "
        "an output directory, call the tool once per CSV file to convert it "
        "to Bronze Parquet. Do not clean, cast, join, or otherwise transform "
        "the data -- that happens in later layers. Report the resulting "
        "Parquet file paths."
    )

    def __init__(self, provider: Optional[str] = None):
        super().__init__(tools=[csv_to_bronze_parquet], provider=provider)

    def execute(self, sttm: STTMDocument, csv_paths: list[str], output_dir: str) -> str:
        result = self.run(
            "Execute this approved Bronze STTM by converting each CSV to Bronze "
            f"Parquet in '{output_dir}'.\n\nApproved STTM rules:\n{sttm.to_execution_json()}\n\n"
            f"CSV files to process: {csv_paths}"
        )
        return result["output"]
