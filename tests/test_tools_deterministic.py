"""
End-to-end test of the mechanical (non-LLM) tooling: profiling, Bronze
conversion, Silver cleansing, Gold joins/flagging/aggregation, the
parameterized data store, and report rendering. Run directly:

    python3 tests/test_tools_deterministic.py

No LLM API key is required -- this exercises exactly the same tool
functions the ReAct agents call, just invoked directly instead of via an
agent's tool-calling loop, so the transformation logic can be validated
independently of any model.
"""

from __future__ import annotations

import io
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.profiler_tools import profile_csv_files  # noqa: E402
from tools.bronze_tools import csv_to_bronze_parquet  # noqa: E402
from tools.silver_tools import (  # noqa: E402
    cast_column_dtype,
    deduplicate_rows,
    fill_nulls,
    standardize_date_format,
)
from tools.gold_tools import (  # noqa: E402
    aggregate_product_quality_metrics,
    flag_quality_issue_keywords,
    join_tables,
)
from tools.reporter_tools import render_html_report  # noqa: E402
from utils.db import GoldDataStore  # noqa: E402
from utils.security import UnsafeIdentifierError, validate_sql_identifier  # noqa: E402

import pandas as pd  # noqa: E402


def main() -> None:
    work = ROOT / "tests" / "_scratch"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)

    sample_dir = ROOT / "sample_data"
    if not (sample_dir / "products.csv").exists():
        subprocess.run([sys.executable, str(sample_dir / "generate_sample_data.py")], check=True)

    csvs = [str(sample_dir / f) for f in ("products.csv", "sales.csv", "reviews.csv")]

    # Profiler
    profile_json = profile_csv_files.invoke({"csv_paths": csvs})
    assert '"tables"' in profile_json
    print("[ok] profiler")

    # Bronze
    bronze_dir = work / "bronze"
    bronze_paths = {}
    for c in csvs:
        p = csv_to_bronze_parquet.invoke({"csv_path": c, "output_dir": str(bronze_dir)})
        bronze_paths[Path(c).stem] = p
    print("[ok] bronze:", bronze_paths)

    # Silver
    silver_dir = work / "silver"
    silver_dir.mkdir(parents=True)
    silver_paths = {}
    for name, p in bronze_paths.items():
        sp = silver_dir / f"{name}_silver.parquet"
        shutil.copy(p, sp)
        silver_paths[name] = str(sp)

    standardize_date_format.invoke({"parquet_path": silver_paths["sales"], "column": "sale_date"})
    fill_nulls.invoke({"parquet_path": silver_paths["sales"], "column": "quantity", "strategy": "median"})
    cast_column_dtype.invoke({"parquet_path": silver_paths["sales"], "column": "quantity", "target_dtype": "int"})
    deduplicate_rows.invoke({"parquet_path": silver_paths["sales"], "subset_columns": None})
    fill_nulls.invoke({"parquet_path": silver_paths["reviews"], "column": "rating", "strategy": "median"})
    print("[ok] silver")

    # Gold
    flag_quality_issue_keywords.invoke(
        {"parquet_path": silver_paths["reviews"], "review_text_column": "review_text"}
    )
    gold_dir = work / "gold_work"
    j1 = join_tables.invoke({
        "left_parquet": silver_paths["sales"], "right_parquet": silver_paths["products"],
        "left_on": "product_id", "right_on": "product_id", "how": "left",
        "output_path": str(gold_dir / "sales_products.parquet"),
    })
    j2 = join_tables.invoke({
        "left_parquet": j1, "right_parquet": silver_paths["reviews"],
        "left_on": "product_id", "right_on": "product_id", "how": "left",
        "output_path": str(gold_dir / "full_join.parquet"),
    })
    final = aggregate_product_quality_metrics.invoke(
        {"joined_parquet": j2, "output_path": str(work / "products_analysis.parquet")}
    )
    df = pd.read_parquet(final)
    assert (df["quality_issue_rate"] >= 0).all() and (df["quality_issue_rate"] <= 1).all()
    assert not df["quality_issue_rate"].isna().any(), "division-by-zero guard failed"
    print("[ok] gold, top product:", df.sort_values("quality_issue_rate", ascending=False).iloc[0]["product_id"])

    # Data store + security guardrails
    store = GoldDataStore(work / "gold.db")
    store.write_dataframe("products_analysis", df)
    top = store.top_n_by_quality_issue_rate(5)
    assert len(top) == 5
    try:
        store.query("DELETE FROM products_analysis")
        raise AssertionError("non-SELECT query should have been rejected")
    except ValueError:
        pass
    try:
        validate_sql_identifier("products_analysis; DROP TABLE x;")
        raise AssertionError("malicious identifier should have been rejected")
    except UnsafeIdentifierError:
        pass
    print("[ok] data store + SQL-injection guardrails")

    # Reporter
    top_json = top.to_json(orient="records")
    report_path = render_html_report.invoke({
        "business_intent": "Identify products with high negative quality feedback",
        "top_products_json": top_json,
        "output_path": str(work / "report.html"),
    })
    assert Path(report_path).exists()
    print("[ok] reporter ->", report_path)

    print("\nAll deterministic tool tests passed.")


if __name__ == "__main__":
    main()
