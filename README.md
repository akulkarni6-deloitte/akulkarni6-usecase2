# IDAMP — Intent-Driven Agentic Medallion Pipeline

Secure, modular, multi-agent pipeline for e-commerce product quality analysis.
Raw `products.csv` / `sales.csv` / `reviews.csv` flow through a Bronze → Silver
→ Gold medallion architecture, driven by single-purpose LangChain ReAct agents,
with mandatory human-in-the-loop (HITL) approval gates between layers.

## Architecture at a glance

```
Streamlit UI (streamlit_app.py)
        │
        ▼
Supervisor (agents/supervisor.py)      -- deterministic phase dispatch
        │
   ┌────┴─────────────────────────────────────────────┐
   ▼            ▼            ▼           ▼            ▼
Profiler   STTM Generator  Bronze     Silver      Gold + Reporter
(profile.json) (per-layer) (CSV→Parquet) (cleanse) (join/flag/aggregate,
                                                     parameterized SQL report)
```

- **Dynamic LLM client** (`utils/llm_factory.py`): factory + strategy pattern.
  Switch providers via `IDAMP_LLM_PROVIDER` (anthropic / openai / groq / gemini)
  with zero changes to agent code.
- **Secrets** (`utils/secrets.py`): one interface, three backends (env vars,
  AWS Secrets Manager, HashiCorp Vault). Keys are never hardcoded or logged.
- **SQL injection prevention** (`utils/db.py`): the Gold/Reporter data layer
  only executes parameterized queries; table names are whitelist-validated
  since identifiers can't be bound as parameters.
- **Prompt-injection defenses** (`utils/security.py`): all business intent and
  review-text content is wrapped as explicit "untrusted data" blocks before
  being placed in any LLM prompt, and common hijack phrasing is flagged.
- **HITL gates** (`pipeline_state.py`, `streamlit_app.py`): the Bronze, Silver,
  and Gold STTMs are rendered as editable tables; nothing executes against
  real data until a human clicks "Approve."

## Setup

```bash
cd idamp
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# edit .env: set IDAMP_LLM_PROVIDER and the matching API key
```

Optional: generate demo data instead of using your own CSVs:

```bash
python3 sample_data/generate_sample_data.py
```

## Run

```bash
streamlit run streamlit_app.py
```

Then in the browser:
1. Enter (or accept the default) business intent, upload the three CSVs, click
   **Start / update run**.
2. **Phase 1** — review the generated data profile and the proposed Bronze
   STTM, edit if needed, click **Approve**.
3. **Phase 2** — Bronze executes automatically; review/edit/approve the
   Silver STTM.
4. **Phase 3** — Silver executes; review/edit/approve the Gold STTM
   (pay attention to the quality-issue keyword logic and aggregation).
5. **Phase 4** — Gold executes, the report is generated automatically
   (no gate). View the ranked `products_analysis` table and `report.html`,
   and download both.

All intermediate state lives under `./pipeline_data/` (git-ignored), so a
run can be resumed if the Streamlit session restarts. Use **Reset pipeline**
in the sidebar to start clean.

## Running without Streamlit (scripted / CI use)

```python
from pathlib import Path
from pipeline_state import PipelineState
from agents.supervisor import Supervisor

state = PipelineState.load_or_create("./pipeline_data")
state.business_intent = "Analyze product sales and customer reviews to identify products with a high rate of negative quality-related feedback."
state.csv_paths = {"products": "sample_data/products.csv",
                    "sales": "sample_data/sales.csv",
                    "reviews": "sample_data/reviews.csv"}
state.save()

sup = Supervisor(state, provider="anthropic")
sup.run_phase1_profile_and_bronze_sttm()
state.approve_gate(1, "auto-approved for CI")
sup.run_phase2_bronze_execute_and_silver_sttm()
state.approve_gate(2, "auto-approved for CI")
sup.run_phase3_silver_execute_and_gold_sttm()
state.approve_gate(3, "auto-approved for CI")
sup.run_phase4_gold_execute_and_report()
```

## Testing without any LLM API key

Every mechanical tool (profiling, Bronze conversion, Silver cleansing, Gold
joins/flagging/aggregation, parameterized queries, report rendering) is a
plain Python function under `tools/` and can be called directly — this is
how the pipeline logic was validated during development, independent of any
LLM call. See `tests/test_tools_deterministic.py` for a runnable example.

## Extending

- **New LLM provider**: add a `LLMClientStrategy` subclass in
  `utils/llm_factory.py` and register it in `_STRATEGIES`.
- **New secrets backend**: add a `SecretsProvider` subclass in
  `utils/secrets.py` and register it in `get_secrets_provider()`.
- **Smarter quality detection**: `tools/gold_tools.py` includes
  `build_llm_sentiment_prompt()` for an optional LLM-backed classification
  pass alongside the default keyword scan — wire it into `GoldAgent`'s tools
  if keyword matching proves too coarse for your review data.
