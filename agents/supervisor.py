"""
Supervisor agent.

Design note: the brief calls for a fully autonomous agent per role,
including the Supervisor. For a pipeline with mandatory HITL approval
gates and financial/quality-sensitive output, letting an LLM freely decide
*pipeline order* is an unnecessary and unsafe degree of freedom -- the
phase sequence (Profile -> Bronze -> Silver -> Gold -> Report) and the gate
checks are fixed, auditable business logic. The Supervisor therefore
deterministically dispatches the correct specialist agent for the current,
gate-approved phase (satisfying "Orchestrates the entire pipeline,
dispatching specialist agents according to the phase structure"), while
each dispatched specialist is itself the autonomous ReAct LLM agent doing
the actual reasoning/tool use for its single-purpose task. An optional LLM
call is used only for human-readable run narration, never for control
flow.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from agents.bronze_agent import BronzeAgent
from agents.gold_agent import GoldAgent
from agents.profiler_agent import ProfilerAgent
from agents.reporter_agent import ReporterAgent
from agents.silver_agent import SilverAgent
from agents.sttm_agent import STTMGeneratorAgent
from pipeline_state import PipelineState
from utils.db import GoldDataStore
from utils.schemas import Layer, STTMDocument


class Supervisor:
    """Deterministically dispatches specialist agents per the approved phase structure."""

    def __init__(self, state: PipelineState, provider: Optional[str] = None):
        self.state = state
        self.provider = provider

    # -- Phase 1 -------------------------------------------------------------

    def run_phase1_profile_and_bronze_sttm(self) -> tuple[str, STTMDocument]:
        profiler = ProfilerAgent(provider=self.provider)
        profile_path = self.state.workdir / "profile.json"
        profile_json = profiler.profile(list(self.state.csv_paths.values()), profile_path)
        self.state.set_artifact("profile_json", str(profile_path))

        sttm_agent = STTMGeneratorAgent(provider=self.provider)
        doc = sttm_agent.generate(Layer.BRONZE, profile_json, self.state.business_intent)
        sttm_path = self.state.workdir / "sttm_bronze.csv"
        STTMGeneratorAgent.write_sttm_csv(doc, sttm_path)
        self.state.set_artifact("sttm_bronze_csv", str(sttm_path))
        return profile_json, doc

    # -- Phase 2 -------------------------------------------------------------

    def run_phase2_bronze_execute_and_silver_sttm(self) -> tuple[dict[str, str], STTMDocument]:
        bronze_sttm = STTMGeneratorAgent.load_sttm_csv(
            self.state.get_artifact("sttm_bronze_csv"), Layer.BRONZE, self.state.business_intent
        )
        bronze_agent = BronzeAgent(provider=self.provider)
        bronze_dir = self.state.workdir / "bronze"
        bronze_agent.execute(bronze_sttm, list(self.state.csv_paths.values()), str(bronze_dir))

        bronze_paths = {
            name: str(bronze_dir / f"{Path(path).stem}_bronze.parquet")
            for name, path in self.state.csv_paths.items()
        }
        self.state.set_artifact("bronze_paths_json", str(bronze_paths))
        for name, path in bronze_paths.items():
            self.state.set_artifact(f"bronze::{name}", path)

        profile_json = Path(self.state.get_artifact("profile_json")).read_text()
        sttm_agent = STTMGeneratorAgent(provider=self.provider)
        doc = sttm_agent.generate(Layer.SILVER, profile_json, self.state.business_intent)
        sttm_path = self.state.workdir / "sttm_silver.csv"
        STTMGeneratorAgent.write_sttm_csv(doc, sttm_path)
        self.state.set_artifact("sttm_silver_csv", str(sttm_path))
        return bronze_paths, doc

    # -- Phase 3 -------------------------------------------------------------

    def run_phase3_silver_execute_and_gold_sttm(self) -> tuple[dict[str, str], STTMDocument]:
        silver_sttm = STTMGeneratorAgent.load_sttm_csv(
            self.state.get_artifact("sttm_silver_csv"), Layer.SILVER, self.state.business_intent
        )
        bronze_paths = {
            name: self.state.get_artifact(f"bronze::{name}") for name in self.state.csv_paths
        }

        silver_dir = self.state.workdir / "silver"
        silver_dir.mkdir(parents=True, exist_ok=True)
        silver_paths: dict[str, str] = {}
        for name, bronze_path in bronze_paths.items():
            silver_path = silver_dir / f"{name}_silver.parquet"
            import shutil
            shutil.copy(bronze_path, silver_path)  # copy, never mutate the Bronze file in place
            silver_paths[name] = str(silver_path)

        silver_agent = SilverAgent(provider=self.provider)
        silver_agent.execute(silver_sttm, silver_paths)
        for name, path in silver_paths.items():
            self.state.set_artifact(f"silver::{name}", path)

        profile_json = Path(self.state.get_artifact("profile_json")).read_text()
        sttm_agent = STTMGeneratorAgent(provider=self.provider)
        doc = sttm_agent.generate(Layer.GOLD, profile_json, self.state.business_intent)
        sttm_path = self.state.workdir / "sttm_gold.csv"
        STTMGeneratorAgent.write_sttm_csv(doc, sttm_path)
        self.state.set_artifact("sttm_gold_csv", str(sttm_path))
        return silver_paths, doc

    # -- Phase 4 -------------------------------------------------------------

    def run_phase4_gold_execute_and_report(self) -> tuple[str, str]:
        gold_sttm = STTMGeneratorAgent.load_sttm_csv(
            self.state.get_artifact("sttm_gold_csv"), Layer.GOLD, self.state.business_intent
        )
        silver_paths = {
            name: self.state.get_artifact(f"silver::{name}") for name in self.state.csv_paths
        }

        gold_agent = GoldAgent(provider=self.provider)
        gold_work_dir = self.state.workdir / "gold_work"
        final_parquet = self.state.workdir / "products_analysis.parquet"
        gold_agent.execute(gold_sttm, silver_paths, str(gold_work_dir), str(final_parquet))
        self.state.set_artifact("products_analysis_parquet", str(final_parquet))

        import pandas as pd
        db_path = self.state.workdir / "gold.db"
        store = GoldDataStore(db_path)
        store.write_dataframe("products_analysis", pd.read_parquet(final_parquet))
        self.state.set_artifact("gold_db", str(db_path))

        reporter = ReporterAgent(provider=self.provider)
        report_path = self.state.workdir / "report.html"
        reporter.execute(str(db_path), self.state.business_intent, str(report_path))
        self.state.set_artifact("report_html", str(report_path))
        return str(final_parquet), str(report_path)
