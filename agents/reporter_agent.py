from __future__ import annotations

from typing import Optional

from agents.base_agent import BaseSpecialistAgent
from tools.reporter_tools import load_top_products_by_quality_issue, render_html_report


class ReporterAgent(BaseSpecialistAgent):
    """Queries the final Gold table via parameterized SQL and writes report.html."""

    system_prompt = (
        "You are the Reporter agent. Use `load_top_products_by_quality_issue` "
        "(which runs a parameterized SQL query -- never write raw SQL "
        "yourself) to get the top products by quality_issue_rate, then call "
        "`render_html_report` with the given business_intent to produce a "
        "concise report.html directly answering the business intent."
    )

    def __init__(self, provider: Optional[str] = None):
        super().__init__(
            tools=[load_top_products_by_quality_issue, render_html_report],
            provider=provider,
        )

    def execute(self, db_path: str, business_intent: str, output_path: str, top_n: int = 5) -> str:
        result = self.run(
            f"Load the top {top_n} products by quality_issue_rate from the "
            f"database at '{db_path}' using the parameterized query tool, then "
            f"render report.html at '{output_path}' for this business intent: "
            f"'{business_intent}'."
        )
        return result["output"]
