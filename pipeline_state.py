"""
Tracks phase progress and Human-in-the-Loop gate approvals for a pipeline
run, persisted to a JSON state file so a Streamlit session (which reruns
the script on every interaction) can resume correctly.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from utils.schemas import PhaseGateStatus

PHASES = [
    {"phase": 1, "name": "Profile & Bronze STTM", "outputs": ["profile.json", "sttm_bronze.csv"]},
    {"phase": 2, "name": "Bronze Execute & Silver STTM", "outputs": ["*_bronze.parquet", "sttm_silver.csv"]},
    {"phase": 3, "name": "Silver Execute & Gold STTM", "outputs": ["*_silver.parquet", "sttm_gold.csv"]},
    {"phase": 4, "name": "Gold Execute & Report", "outputs": ["products_analysis.parquet", "report.html"]},
]


@dataclass
class PipelineState:
    workdir: Path
    business_intent: str = ""
    csv_paths: dict[str, str] = field(default_factory=dict)
    gates: dict[int, PhaseGateStatus] = field(default_factory=dict)
    artifacts: dict[str, str] = field(default_factory=dict)  # logical name -> path

    def __post_init__(self) -> None:
        if not self.gates:
            self.gates = {
                p["phase"]: PhaseGateStatus(phase=p["phase"], name=p["name"], outputs=p["outputs"])
                for p in PHASES
            }

    # -- persistence -------------------------------------------------------

    @property
    def state_path(self) -> Path:
        return self.workdir / "pipeline_state.json"

    def save(self) -> None:
        self.workdir.mkdir(parents=True, exist_ok=True)
        payload = {
            "business_intent": self.business_intent,
            "csv_paths": self.csv_paths,
            "gates": {str(k): v.model_dump() for k, v in self.gates.items()},
            "artifacts": self.artifacts,
        }
        self.state_path.write_text(json.dumps(payload, indent=2))

    @classmethod
    def load_or_create(cls, workdir: str | Path) -> "PipelineState":
        workdir = Path(workdir)
        state = cls(workdir=workdir)
        if state.state_path.exists():
            payload = json.loads(state.state_path.read_text())
            state.business_intent = payload.get("business_intent", "")
            state.csv_paths = payload.get("csv_paths", {})
            state.artifacts = payload.get("artifacts", {})
            gates = payload.get("gates", {})
            state.gates = {
                int(k): PhaseGateStatus.model_validate(v) for k, v in gates.items()
            } or state.gates
        return state

    # -- gate helpers --------------------------------------------------------

    def current_phase(self) -> int:
        for phase_num in sorted(self.gates):
            if not self.gates[phase_num].approved:
                return phase_num
        return len(self.gates) + 1  # all approved -> complete

    def approve_gate(self, phase: int, note: str = "") -> None:
        self.gates[phase].approved = True
        self.gates[phase].approver_note = note
        self.save()

    def reject_gate(self, phase: int, note: str = "") -> None:
        self.gates[phase].approved = False
        self.gates[phase].approver_note = note
        self.save()

    def set_artifact(self, name: str, path: str) -> None:
        self.artifacts[name] = path
        self.save()

    def get_artifact(self, name: str) -> Optional[str]:
        return self.artifacts.get(name)
