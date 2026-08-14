"""
IDAMP Streamlit UI.

Drives the four-phase pipeline with mandatory Human-in-the-Loop approval
gates between phases. Each phase's specialist agents run only after the
user clicks through this app; the STTM proposed by the LLM at each layer
is shown as an editable table so the human can correct/approve it before
any code execution against real data.
"""

from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import streamlit as st

from agents.supervisor import Supervisor
from pipeline_state import PipelineState
from utils.schemas import Layer

st.set_page_config(page_title="IDAMP - Product Quality Pipeline", layout="wide")

WORKDIR = Path("./pipeline_data")


def get_state() -> PipelineState:
    if "state" not in st.session_state:
        st.session_state.state = PipelineState.load_or_create(WORKDIR)
    return st.session_state.state


def sidebar_setup(state: PipelineState) -> str:
    st.sidebar.header("Configuration")
    provider = st.sidebar.selectbox(
        "LLM Provider", ["anthropic", "openai", "groq", "gemini"], index=0,
        help="API keys are read from environment variables / secrets backend, never entered here.",
    )
    st.sidebar.caption("Keys are managed via `.env` / your secrets backend "
                        "(see `.env.example`) -- never pasted into the UI.")
    st.sidebar.divider()
    st.sidebar.subheader("Pipeline status")
    for phase_num in sorted(state.gates):
        gate = state.gates[phase_num]
        icon = "✅" if gate.approved else ("🟡" if phase_num == state.current_phase() else "⬜")
        st.sidebar.write(f"{icon} Phase {gate.phase}: {gate.name}")
    if st.sidebar.button("Reset pipeline", type="secondary"):
        for f in WORKDIR.glob("**/*"):
            if f.is_file():
                f.unlink()
        st.session_state.pop("state", None)
        st.rerun()
    return provider


def render_intent_and_uploads(state: PipelineState) -> bool:
    st.header("1. Business intent & source data")
    intent = st.text_area(
        "Business intent", value=state.business_intent or
        "Analyze product sales and customer reviews to identify products with "
        "a high rate of negative quality-related feedback.",
        height=80,
    )
    col1, col2, col3 = st.columns(3)
    uploads = {}
    with col1:
        uploads["products"] = st.file_uploader("products.csv", type="csv", key="products")
    with col2:
        uploads["sales"] = st.file_uploader("sales.csv", type="csv", key="sales")
    with col3:
        uploads["reviews"] = st.file_uploader("reviews.csv", type="csv", key="reviews")

    ready = intent and all(uploads.values())
    if st.button("Start / update run", disabled=not ready, type="primary"):
        state.business_intent = intent
        raw_dir = state.workdir / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        for name, uploaded in uploads.items():
            dest = raw_dir / f"{name}.csv"
            dest.write_bytes(uploaded.getvalue())
            state.csv_paths[name] = str(dest)
        state.save()
        st.success("Inputs saved. Proceed to Phase 1 below.")
        st.rerun()

    if not ready:
        st.info("Upload all three CSVs and provide a business intent to begin.")
    return bool(state.csv_paths)


def editable_sttm(csv_path: str, key: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    st.caption("Review and edit the proposed rules below before approving this gate.")
    edited = st.data_editor(df, num_rows="dynamic", use_container_width=True, key=key)
    return edited


def save_edited_sttm(df: pd.DataFrame, csv_path: str) -> None:
    # st.data_editor's "dynamic" row mode can leave a fully-blank row behind;
    # drop rows with no target_table rather than persist an empty rule.
    if "target_table" in df.columns:
        df = df[df["target_table"].notna() & (df["target_table"].astype(str).str.strip() != "")]
    df.to_csv(csv_path, index=False)


def phase_gate_ui(state: PipelineState, phase_num: int, supervisor: Supervisor) -> None:
    gate = state.gates[phase_num]
    st.header(f"Phase {phase_num}: {gate.name}")

    if phase_num == 1:
        if not state.get_artifact("sttm_bronze_csv"):
            with st.spinner("Profiling raw CSVs and generating Bronze STTM..."):
                supervisor.run_phase1_profile_and_bronze_sttm()
        st.subheader("Data profile")
        st.json(Path(state.get_artifact("profile_json")).read_text())
        st.subheader("Proposed Bronze STTM")
        df = editable_sttm(state.get_artifact("sttm_bronze_csv"), "bronze_sttm_editor")
        gate_col1, gate_col2 = st.columns(2)
        note = gate_col1.text_input("Approver note", key="note_p1")
        if gate_col2.button("Approve Bronze STTM & continue", type="primary", key="approve_p1"):
            save_edited_sttm(df, state.get_artifact("sttm_bronze_csv"))
            state.approve_gate(1, note)
            st.rerun()

    elif phase_num == 2:
        if not state.get_artifact("sttm_silver_csv"):
            with st.spinner("Executing Bronze layer and generating Silver STTM..."):
                supervisor.run_phase2_bronze_execute_and_silver_sttm()
        st.subheader("Bronze Parquet files produced")
        st.write({k: v for k, v in state.artifacts.items() if k.startswith("bronze::")})
        st.subheader("Proposed Silver STTM")
        df = editable_sttm(state.get_artifact("sttm_silver_csv"), "silver_sttm_editor")
        gate_col1, gate_col2 = st.columns(2)
        note = gate_col1.text_input("Approver note", key="note_p2")
        if gate_col2.button("Approve Silver STTM & continue", type="primary", key="approve_p2"):
            save_edited_sttm(df, state.get_artifact("sttm_silver_csv"))
            state.approve_gate(2, note)
            st.rerun()

    elif phase_num == 3:
        if not state.get_artifact("sttm_gold_csv"):
            with st.spinner("Executing Silver layer and generating Gold STTM..."):
                supervisor.run_phase3_silver_execute_and_gold_sttm()
        st.subheader("Silver Parquet files produced")
        st.write({k: v for k, v in state.artifacts.items() if k.startswith("silver::")})
        st.subheader("Proposed Gold STTM")
        st.warning("Pay close attention to the quality-issue detection logic and aggregation metrics.")
        df = editable_sttm(state.get_artifact("sttm_gold_csv"), "gold_sttm_editor")
        gate_col1, gate_col2 = st.columns(2)
        note = gate_col1.text_input("Approver note", key="note_p3")
        if gate_col2.button("Approve Gold STTM & continue", type="primary", key="approve_p3"):
            save_edited_sttm(df, state.get_artifact("sttm_gold_csv"))
            state.approve_gate(3, note)
            st.rerun()

    elif phase_num == 4:
        if not state.get_artifact("report_html"):
            with st.spinner("Executing Gold layer and generating report..."):
                supervisor.run_phase4_gold_execute_and_report()
            state.approve_gate(4, "auto - no gate required")
        st.success("Pipeline complete.")
        final_df = pd.read_parquet(state.get_artifact("products_analysis_parquet"))
        st.subheader("Gold table: products_analysis")
        st.dataframe(final_df.sort_values("quality_issue_rate", ascending=False), use_container_width=True)

        report_html = Path(state.get_artifact("report_html")).read_text()
        st.subheader("report.html preview")
        st.components.v1.html(report_html, height=500, scrolling=True)
        st.download_button(
            "Download report.html", data=report_html,
            file_name="report.html", mime="text/html",
        )
        buf = io.BytesIO()
        final_df.to_parquet(buf, index=False)
        st.download_button(
            "Download products_analysis.parquet", data=buf.getvalue(),
            file_name="products_analysis.parquet", mime="application/octet-stream",
        )


def main() -> None:
    st.title("IDAMP — Intent-Driven Agentic Medallion Pipeline")
    st.caption("E-commerce product quality analysis, with mandatory human approval "
               "gates between medallion layers.")

    state = get_state()
    provider = sidebar_setup(state)
    supervisor = Supervisor(state, provider=provider)

    has_inputs = render_intent_and_uploads(state)
    if not has_inputs:
        return

    st.divider()
    current = state.current_phase()
    if current > len(state.gates):
        st.success("All phases complete. Scroll down for the final report, or expand earlier phases below.")
        current = len(state.gates)

    for phase_num in range(1, current + 1):
        with st.expander(f"Phase {phase_num}", expanded=(phase_num == current)):
            phase_gate_ui(state, phase_num, supervisor)


if __name__ == "__main__":
    main()
