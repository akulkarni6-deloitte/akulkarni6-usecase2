from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from typing import Optional

from utils.llm_factory import LLMClientFactory
from utils.schemas import Layer, STTMDocument
from utils.security import wrap_untrusted

_LAYER_GUIDANCE = {
    Layer.BRONZE: (
        "Bronze STTM rules should only describe: CSV-to-Parquet ingestion and "
        "metadata injection (load timestamp, source file). No cleansing or "
        "business logic belongs at this layer."
    ),
    Layer.SILVER: (
        "Silver STTM rules should describe cleansing/standardization only: "
        "null handling (mean/median/constant/mode), dtype casting, date format "
        "standardization to YYYY-MM-DD, de-duplication, and surrogate keys if "
        "needed. No joins or aggregations belong at this layer."
    ),
    Layer.GOLD: (
        "Gold STTM rules should describe: joins across the cleaned Silver "
        "tables, the quality_issue_flag derivation from review_text keywords, "
        "and aggregation per product_id producing total_sales, average_rating, "
        "and quality_issue_rate (guard division by zero)."
    ),
}


class STTMGeneratorAgent:
    """
    Unified STTM Generator. Takes a data profile + business intent and
    proposes a structured, validated STTMDocument for a specific layer.
    Uses the LLM's structured-output mode so the result is parsed into
    Pydantic models (never trusted as raw/free-form text).
    """

    def __init__(self, provider: Optional[str] = None):
        self.llm = LLMClientFactory.get_client(provider=provider, temperature=0.0)
        self.structured_llm = self.llm.with_structured_output(STTMDocument)

    def generate(self, layer: Layer, profile_json: str, business_intent: str) -> STTMDocument:
        guidance = _LAYER_GUIDANCE[layer]
        prompt = (
            f"You are the STTM Generator agent, producing the {layer.value.upper()} "
            f"Source-to-Target Mapping for a medallion data pipeline.\n\n"
            f"{guidance}\n\n"
            f"{wrap_untrusted('business_intent', business_intent, max_len=400)}\n\n"
            f"{wrap_untrusted('data_profile_json', profile_json, max_len=2500)}\n\n"
            f"Return a structured STTMDocument with layer='{layer.value}', "
            f"business_intent set to the given intent, and a concise, complete "
            f"list of STTMRule entries. Be precise and avoid unnecessary rules."
        )
        doc = self.structured_llm.invoke(prompt)
        if not isinstance(doc, STTMDocument):
            doc = STTMDocument.model_validate(doc)
        return doc

    @staticmethod
    def to_csv(doc: STTMDocument) -> str:
        buf = io.StringIO()
        writer = csv.DictWriter(
            buf,
            fieldnames=["target_table", "source_table", "source_column",
                        "target_column", "transformation", "rationale"],
        )
        writer.writeheader()
        for rule in doc.rules:
            writer.writerow(rule.model_dump())
        return buf.getvalue()

    @staticmethod
    def write_sttm_csv(doc: STTMDocument, out_path: str | Path) -> Path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(STTMGeneratorAgent.to_csv(doc))
        return out_path

    @staticmethod
    def load_sttm_csv(path: str | Path, layer: Layer, business_intent: str) -> STTMDocument:
        """Reload a (possibly human-edited) STTM CSV after HITL approval."""
        from utils.schemas import STTMRule

        optional_fields = {"source_column", "target_column"}
        required_str_fields = {"target_table", "source_table", "transformation", "rationale"}

        rules = []
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                data = dict(row)
                if not (data.get("target_table") or "").strip():
                    continue  # skip fully-blank/placeholder rows
                for field in optional_fields:
                    if not data.get(field):
                        data[field] = None
                for field in required_str_fields:
                    if data.get(field) is None:
                        data[field] = ""
                rules.append(STTMRule(**data))
        return STTMDocument(layer=layer, business_intent=business_intent, rules=rules)
