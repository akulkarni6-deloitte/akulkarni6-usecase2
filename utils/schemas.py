"""Structured, validated data contracts passed between agents.

Using Pydantic models (instead of free-form dicts/strings) for agent I/O
gives us: (1) validation of LLM-produced JSON before it is trusted anywhere
downstream, (2) a stable contract each agent can rely on, (3) automatic
JSON-schema generation we hand to the LLM for structured output.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class Layer(str, Enum):
    BRONZE = "bronze"
    SILVER = "silver"
    GOLD = "gold"


# ---------------------------------------------------------------------------
# Profiler output
# ---------------------------------------------------------------------------

class ColumnProfile(BaseModel):
    name: str
    inferred_dtype: str
    null_count: int
    distinct_count: Optional[int] = None
    min_value: Optional[Any] = None
    max_value: Optional[Any] = None
    mean_value: Optional[float] = None
    sample_values: list[Any] = Field(default_factory=list)


class TableProfile(BaseModel):
    file_name: str
    row_count: int
    columns: list[ColumnProfile]


class DataProfile(BaseModel):
    tables: list[TableProfile]


# ---------------------------------------------------------------------------
# STTM (Source-to-Target Mapping) output
# ---------------------------------------------------------------------------

class STTMRule(BaseModel):
    target_table: str
    source_table: str
    source_column: Optional[str] = None
    target_column: Optional[str] = None
    transformation: str = Field(
        description="Human + machine readable transformation, e.g. "
                    "'cast:datetime', 'fillna:mean', 'dedupe', "
                    "'join:products.product_id=sales.product_id', "
                    "'derive:quality_issue_flag'"
    )
    rationale: str = ""


class STTMDocument(BaseModel):
    layer: Layer
    business_intent: str
    rules: list[STTMRule]

    def to_dataframe_records(self) -> list[dict]:
        return [r.model_dump() for r in self.rules]

    def to_execution_json(self) -> str:
        """
        Compact JSON for LLM-facing execution prompts: drops `rationale`
        (a human-review field, irrelevant to which tool an agent should
        call) to keep the Bronze/Silver/Gold agent prompts small.
        """
        import json
        rules = [
            {k: v for k, v in r.model_dump().items() if k != "rationale"}
            for r in self.rules
        ]
        return json.dumps({"layer": self.layer.value, "rules": rules}, separators=(",", ":"))


# ---------------------------------------------------------------------------
# Gold-layer aggregate result
# ---------------------------------------------------------------------------

class ProductQualityMetric(BaseModel):
    product_id: str
    product_name: str
    category: str
    total_sales: float
    units_sold: int
    average_rating: Optional[float] = None
    review_count: int
    negative_quality_reviews: int
    quality_issue_rate: float


class PhaseGateStatus(BaseModel):
    phase: int
    name: str
    outputs: list[str] = Field(default_factory=list)
    approved: bool = False
    approver_note: str = ""
