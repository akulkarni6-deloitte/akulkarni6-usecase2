from __future__ import annotations

from typing import Optional

from agents.base_agent import BaseSpecialistAgent
from tools.gold_tools import (
    aggregate_product_quality_metrics,
    flag_quality_issue_keywords,
    join_tables,
)
from utils.schemas import STTMDocument


class GoldAgent(BaseSpecialistAgent):
    """Executes the approved Gold STTM: joins, quality flagging, aggregation."""

    system_prompt = (
        "You are the Gold agent in a medallion data pipeline. Using your "
        "tools (join_tables, flag_quality_issue_keywords, "
        "aggregate_product_quality_metrics), execute the approved Gold STTM: "
        "join the Silver tables, derive quality_issue_flag from review_text "
        "keywords, then aggregate per product_id into total_sales, "
        "units_sold, average_rating, review_count, negative_quality_reviews, "
        "and quality_issue_rate. The aggregation tool already guards against "
        "division-by-zero -- do not attempt manual arithmetic yourself, "
        "always use the tools."
    )
    max_iterations = 8

    def __init__(self, provider: Optional[str] = None):
        super().__init__(
            tools=[join_tables, flag_quality_issue_keywords, aggregate_product_quality_metrics],
            provider=provider,
        )

    def execute(self, sttm: STTMDocument, silver_parquet_paths: dict[str, str],
                work_dir: str, final_output_path: str) -> str:
        result = self.run(
            "Execute this approved Gold STTM against the Silver Parquet files "
            f"below. Use intermediate files inside '{work_dir}' as needed, and "
            f"write the final aggregated table to '{final_output_path}'.\n\n"
            f"Approved STTM rules:\n{sttm.to_execution_json()}\n\n"
            f"Silver parquet paths by table: {silver_parquet_paths}"
        )
        return result["output"]
