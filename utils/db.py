"""
Data store access layer used by the Gold Agent and Reporter agent.

Hard rule (per project brief, section 3): every query that touches user-
supplied or LLM-inferred values MUST use parameterized queries (`?`
placeholders bound via `execute(sql, params)`), never Python string
formatting/f-strings with values spliced into the SQL text.

Table and column *names* cannot be bound as bind-parameters in SQL, so
those are validated against a whitelist (`utils.security.validate_sql_identifier`)
before being placed in the SQL string. Only identifiers pass through this
path; values always go through the parameter list.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator

import pandas as pd

from utils.security import validate_sql_identifier

ALLOWED_TABLES = {"products_analysis", "products", "sales", "reviews"}


class GoldDataStore:
    """Thin, safe wrapper around SQLite for the Gold layer and Reporter."""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def write_dataframe(self, table_name: str, df: pd.DataFrame) -> None:
        """Persist a DataFrame to a whitelisted table (schema-managed, replace mode)."""
        validate_sql_identifier(table_name, allowed=ALLOWED_TABLES)
        with self._connect() as conn:
            df.to_sql(table_name, conn, if_exists="replace", index=False)

    def query(self, sql: str, params: Iterable[Any] = ()) -> pd.DataFrame:
        """
        Execute a parameterized SELECT. `sql` must contain only `?`
        placeholders for values -- never interpolate values into `sql`
        directly. Raises if the statement looks like it isn't a SELECT,
        as a defense-in-depth guard for this read-only accessor.
        """
        normalized = sql.strip().lower()
        if not normalized.startswith("select"):
            raise ValueError("GoldDataStore.query only permits SELECT statements.")
        with self._connect() as conn:
            return pd.read_sql_query(sql, conn, params=tuple(params))

    def top_n_by_quality_issue_rate(self, n: int = 5) -> pd.DataFrame:
        """
        Example of the required pattern: identifier (table name) is
        whitelist-validated and placed in the SQL text; the only *value*
        (`n`) is bound as a parameter, never string-formatted in.
        """
        table = validate_sql_identifier("products_analysis", allowed=ALLOWED_TABLES)
        sql = (
            f"SELECT product_id, product_name, category, total_sales, units_sold, "
            f"average_rating, review_count, negative_quality_reviews, quality_issue_rate "
            f"FROM {table} "
            f"ORDER BY quality_issue_rate DESC "
            f"LIMIT ?"
        )
        return self.query(sql, params=(n,))
