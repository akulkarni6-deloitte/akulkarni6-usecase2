"""Gold Agent's tools: joins, quality-issue detection, and aggregation.

The quality_issue_flag defaults to a fast, deterministic keyword scan
(per the brief's example: "broken", "stopped working", "poor quality").
An optional LLM-backed sentiment pass is also exposed for cases where the
agent judges keyword matching insufficient (sarcasm, negation, etc.) --
the LLM call goes through the same untrusted-content wrapping as any other
review text.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from langchain_core.tools import tool

from utils.security import wrap_untrusted

DEFAULT_QUALITY_KEYWORDS = [
    "broken", "stopped working", "poor quality", "defective", "malfunction",
    "cheaply made", "fell apart", "doesn't work", "does not work", "faulty",
    "damaged", "cracked", "low quality", "wore out", "stopped functioning",
]


def _load(parquet_path: str) -> pd.DataFrame:
    return pd.read_parquet(parquet_path)


def _save(df: pd.DataFrame, parquet_path: str) -> str:
    out_path = Path(parquet_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    return str(out_path)


@tool
def join_tables(left_parquet: str, right_parquet: str, left_on: str, right_on: str,
                 how: str, output_path: str) -> str:
    """
    Join two Silver Parquet tables (`left_parquet`, `right_parquet`) on
    `left_on`/`right_on` using join type `how` ('inner', 'left', 'right',
    'outer'). Writes the joined result to `output_path` and returns it.
    """
    left = _load(left_parquet)
    right = _load(right_parquet)
    merged = left.merge(right, left_on=left_on, right_on=right_on, how=how, suffixes=("_l", "_r"))
    return _save(merged, output_path)


@tool
def flag_quality_issue_keywords(parquet_path: str, review_text_column: str,
                                 flag_column: str = "quality_issue_flag",
                                 keywords: list[str] | None = None) -> str:
    """
    Add a boolean `flag_column` to the Parquet file: True if
    `review_text_column` contains any quality-issue keyword (default list
    covers phrases like 'broken', 'stopped working', 'poor quality';
    override with `keywords`). Review text is treated as untrusted data --
    only substring matching is performed, it is never executed or used to
    alter control flow. Overwrites the file in place and returns its path.
    """
    df = _load(parquet_path)
    if review_text_column not in df.columns:
        return f"ERROR: column '{review_text_column}' not found in {parquet_path}"

    kw_list = [k.lower() for k in (keywords or DEFAULT_QUALITY_KEYWORDS)]
    text_series = df[review_text_column].fillna("").astype(str).str.lower()
    df[flag_column] = text_series.apply(lambda t: any(k in t for k in kw_list))
    return _save(df, parquet_path)


def build_llm_sentiment_prompt(review_text: str) -> str:
    """
    Builds a prompt for optional LLM-backed quality-issue classification.
    Review text is wrapped as explicitly untrusted data so the model
    treats it as content to classify, never as instructions to follow.
    """
    data_block = wrap_untrusted("customer_review_text", review_text)
    return (
        "Classify whether the following customer review describes a "
        "PRODUCT QUALITY problem (defects, breakage, malfunction, poor "
        "build quality) as opposed to shipping, price, or customer-service "
        "complaints. Respond with only 'true' or 'false'.\n\n"
        f"{data_block}"
    )


@tool
def aggregate_product_quality_metrics(joined_parquet: str, output_path: str,
                                       product_id_col: str = "product_id",
                                       product_name_col: str = "product_name",
                                       category_col: str = "category",
                                       quantity_col: str = "quantity",
                                       price_col: str = "price_per_item",
                                       rating_col: str = "rating",
                                       flag_col: str = "quality_issue_flag") -> str:
    """
    Aggregate the joined Gold table per product_id into: total_sales,
    units_sold, average_rating, review_count, negative_quality_reviews, and
    quality_issue_rate (negative_quality_reviews / review_count, safely
    handling division-by-zero as 0.0 when a product has no reviews).
    Writes the result to `output_path` and returns its path.
    """
    df = _load(joined_parquet)

    df["_line_total"] = pd.to_numeric(df[quantity_col], errors="coerce").fillna(0) * \
        pd.to_numeric(df[price_col], errors="coerce").fillna(0)

    grouped = df.groupby([product_id_col, product_name_col, category_col], dropna=False)

    agg = grouped.agg(
        total_sales=("_line_total", "sum"),
        units_sold=(quantity_col, lambda s: pd.to_numeric(s, errors="coerce").fillna(0).sum()),
        average_rating=(rating_col, "mean"),
        review_count=(rating_col, "count"),
        negative_quality_reviews=(flag_col, "sum"),
    ).reset_index()

    agg["quality_issue_rate"] = agg.apply(
        lambda row: (row["negative_quality_reviews"] / row["review_count"])
        if row["review_count"] > 0 else 0.0,
        axis=1,
    )

    agg = agg.rename(columns={
        product_id_col: "product_id",
        product_name_col: "product_name",
        category_col: "category",
    })

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    agg.to_parquet(out_path, index=False)
    return str(out_path)
