"""Pydantic models for DataForge-Env action space, observation space, and rewards."""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Action Space
# ---------------------------------------------------------------------------

class Strategy(str, Enum):
    MEAN = "mean"
    MEDIAN = "median"
    MODE = "mode"
    CONSTANT = "constant"
    DROP = "drop"


class ActionType(str, Enum):
    FILL_MISSING = "fill_missing"
    DROP_DUPLICATES = "drop_duplicates"
    CAST_TYPE = "cast_type"
    NORMALIZE = "normalize"
    JOIN = "join"
    VALIDATE = "validate"


class Action(BaseModel):
    action_type: ActionType = Field(description="The type of cleaning action to perform.")
    params: Dict[str, Any] = Field(
        default_factory=dict,
        description="Action-specific parameters (e.g. column, strategy, key).",
    )


# ---------------------------------------------------------------------------
# Observation Space
# ---------------------------------------------------------------------------

class ColumnStats(BaseModel):
    dtype: str
    null_count: int
    unique_count: int
    mean: Optional[float] = None
    outliers_detected: int = 0


class Observation(BaseModel):
    dataset_preview: List[Dict[str, Any]] = Field(
        description="Top-5 rows of the current dataframe.",
    )
    schema_info: Dict[str, ColumnStats] = Field(
        description="Per-column statistics.",
    )
    validation_errors: List[str] = Field(default_factory=list)
    action_history: List[str] = Field(default_factory=list)
    current_step: int = 0
    max_steps: int = 15
    progress_score: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Current cleanliness score [0-1].",
    )
    dataset_size: int = 0
    progress_delta: float = 0.0


# ---------------------------------------------------------------------------
# Reward Model
# ---------------------------------------------------------------------------

class Reward(BaseModel):
    scalar: float = Field(description="The final normalised reward in [0,1].")
    components: Dict[str, float] = Field(
        default_factory=dict,
        description="Breakdown by component (schema, nulls, dupes, logic, step_penalty).",
    )
    reasoning: str = ""


# ---------------------------------------------------------------------------
# Step Result
# ---------------------------------------------------------------------------

class StepResult(BaseModel):
    observation: Observation
    reward: Reward
    done: bool = False
    info: Dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# API Request / Response helpers
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    task_id: str = Field(default="easy", description="One of: easy, medium, hard.")


class ResetResponse(BaseModel):
    observation: Observation
    info: Dict[str, Any] = Field(default_factory=dict)


class StepRequest(BaseModel):
    action: Action
