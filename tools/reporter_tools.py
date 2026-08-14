"""Reporter agent's tools.

All data-store access goes through `utils.db.GoldDataStore`, which enforces
parameterized queries. This module never builds SQL by string-formatting
values into a query.
"""

from __future__ import annotations

import html
import io
from pathlib import Path

import pandas as pd
from langchain_core.tools import tool

from utils.db import GoldDataStore


@tool
def load_top_products_by_quality_issue(db_path: str, top_n: int = 5) -> str:
    """
    Query the Gold data store (parameterized SQL only) for the products
    with the highest quality_issue_rate, limited to `top_n`. Returns the
    result as a JSON string of records.
    """
    store = GoldDataStore(db_path)
    df = store.top_n_by_quality_issue_rate(n=top_n)
    return df.to_json(orient="records")


@tool
def render_html_report(business_intent: str, top_products_json: str, output_path: str) -> str:
    """
    Render report.html summarizing the top quality-issue products for the
    given business_intent. `top_products_json` should be the JSON records
    string from `load_top_products_by_quality_issue`. All values are
    HTML-escaped before insertion. Returns the output path.
    """
    df = pd.read_json(io.StringIO(top_products_json))

    rows_html = ""
    for _, row in df.iterrows():
        rows_html += "<tr>" + "".join(
            f"<td>{html.escape(str(row.get(col, '')))}</td>"
            for col in ["product_id", "product_name", "category", "total_sales",
                        "units_sold", "average_rating", "review_count",
                        "negative_quality_reviews", "quality_issue_rate"]
        ) + "</tr>\n"

    doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Product Quality Analysis Report</title>
<style>
  body {{ font-family: -apple-system, Segoe UI, Roboto, sans-serif; margin: 2rem; color: #1a1a1a; }}
  h1 {{ font-size: 1.4rem; }}
  .intent {{ color: #555; margin-bottom: 1.5rem; }}
  table {{ border-collapse: collapse; width: 100%; }}
  th, td {{ border: 1px solid #ddd; padding: 8px 10px; text-align: left; font-size: 0.9rem; }}
  th {{ background: #f4f4f4; }}
  tr:nth-child(even) {{ background: #fafafa; }}
</style>
</head>
<body>
  <h1>Product Quality Analysis Report</h1>
  <p class="intent"><strong>Business intent:</strong> {html.escape(business_intent)}</p>
  <table>
    <thead>
      <tr>
        <th>Product ID</th><th>Product Name</th><th>Category</th>
        <th>Total Sales</th><th>Units Sold</th><th>Avg Rating</th>
        <th>Review Count</th><th>Negative Quality Reviews</th><th>Quality Issue Rate</th>
      </tr>
    </thead>
    <tbody>
      {rows_html}
    </tbody>
  </table>
</body>
</html>
"""
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(doc)
    return str(out_path)
