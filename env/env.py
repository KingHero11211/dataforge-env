"""DataForge-Env – stateful data-cleaning environment.

Provides async-capable reset / step / state methods used by the FastAPI server
and the inference script.
"""

from __future__ import annotations

import copy
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from env.graders import grade_task, normalize_score
from env.models import (
    Action,
    ActionType,
    ColumnStats,
    Observation,
    Reward,
    StepResult,
)
from env.tasks import TaskDefinition, get_task


class DataForgeEnv:
    """Stateful environment wrapping a data-cleaning episode."""

    def __init__(self) -> None:
        self._task: Optional[TaskDefinition] = None
        self._dataframes: Dict[str, pd.DataFrame] = {}
        self._current_step: int = 0
        self._action_history: List[str] = []
        self._done: bool = False
        self._prev_score: float = 0.0

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    async def reset(self, task_id: str = "easy") -> Observation:
        """Initialise a fresh episode for *task_id* and return the first observation."""
        self._task = get_task(task_id)
        self._dataframes = {
            k: v.copy() for k, v in self._task.dirty_dataframes.items()
        }
        self._current_step = 0
        self._action_history = []
        self._done = False
        self._prev_score = 0.0
        return self._build_observation()

    async def step(self, action: Action) -> StepResult:
        """Apply *action* and return (observation, reward, done, info)."""
        if self._task is None:
            raise RuntimeError("Environment not initialised. Call reset() first.")
        if self._done:
            obs = self._build_observation()
            return StepResult(
                observation=obs,
                reward=Reward(scalar=normalize_score(0.01), components={}, reasoning="Episode already finished."),
                done=True,
                info={"warning": "Episode already done."},
            )

        self._current_step += 1
        error_msg = self._apply_action(action)

        # Compute reward
        reward = self._compute_reward(error_msg)

        # Check termination
        if self._current_step >= self._task.max_steps:
            self._done = True

        obs = self._build_observation()

        return StepResult(
            observation=obs,
            reward=reward,
            done=self._done,
            info={"error": error_msg} if error_msg else {},
        )

    def state(self) -> Dict[str, Any]:
        """Return a JSON-serialisable snapshot of the environment state."""
        if self._task is None:
            return {"initialised": False}
        preview: Dict[str, Any] = {}
        for name, df in self._dataframes.items():
            preview[name] = {
                "shape": list(df.shape),
                "columns": list(df.columns),
                "head": df.head(5).fillna("null").to_dict(orient="records"),
            }
        return {
            "initialised": True,
            "task_id": self._task.task_id,
            "task_name": self._task.name,
            "current_step": self._current_step,
            "max_steps": self._task.max_steps,
            "done": self._done,
            "action_history": list(self._action_history),
            "dataframes": preview,
        }

    # ------------------------------------------------------------------
    # Action dispatch
    # ------------------------------------------------------------------

    def _apply_action(self, action: Action) -> Optional[str]:
        """Dispatch *action* to the correct handler.  Returns error string or None."""
        try:
            handler = {
                ActionType.FILL_MISSING: self._act_fill_missing,
                ActionType.DROP_DUPLICATES: self._act_drop_duplicates,
                ActionType.CAST_TYPE: self._act_cast_type,
                ActionType.NORMALIZE: self._act_normalize,
                ActionType.JOIN: self._act_join,
                ActionType.VALIDATE: self._act_validate,
            }.get(action.action_type)

            if handler is None:
                return f"Unknown action_type: {action.action_type}"

            self._action_history.append(
                f"step={self._current_step} action={action.action_type.value} params={action.params}"
            )
            return handler(action.params)
        except Exception as exc:
            return f"Action error: {exc}"

    # --- Individual handlers ---------------------------------------------------

    def _get_df(self, params: Dict[str, Any], key: str = "main") -> pd.DataFrame:
        table = params.get("table", key)
        if table not in self._dataframes:
            raise KeyError(f"Table '{table}' not found. Available: {list(self._dataframes.keys())}")
        return self._dataframes[table]

    def _act_fill_missing(self, params: Dict[str, Any]) -> Optional[str]:
        df = self._get_df(params)
        col = params.get("column")
        if col is None:
            return "Missing required param 'column'."
        if col not in df.columns:
            return f"Column '{col}' does not exist. Available: {list(df.columns)}"
        strategy = params.get("strategy", "drop")

        if strategy == "mean":
            numeric = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].fillna(numeric.mean())
        elif strategy == "median":
            numeric = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].fillna(numeric.median())
        elif strategy == "mode":
            mode_val = df[col].mode()
            if len(mode_val) > 0:
                df[col] = df[col].fillna(mode_val.iloc[0])
        elif strategy == "constant":
            fill_value = params.get("fill_value", "")
            df[col] = df[col].fillna(fill_value)
        elif strategy == "drop":
            before = len(df)
            df = df.dropna(subset=[col]).reset_index(drop=True)
            if len(df) == 0:
                return "Destructive action: all rows dropped."
            self._dataframes[params.get("table", "main")] = df
        else:
            return f"Unknown strategy '{strategy}'."
        return None

    def _act_drop_duplicates(self, params: Dict[str, Any]) -> Optional[str]:
        df = self._get_df(params)
        subset = params.get("subset")
        before = len(df)
        if subset:
            bad_cols = [c for c in subset if c not in df.columns]
            if bad_cols:
                return f"Columns {bad_cols} not found."
            df = df.drop_duplicates(subset=subset).reset_index(drop=True)
        else:
            df = df.drop_duplicates().reset_index(drop=True)
        if len(df) == 0:
            return "Destructive action: all rows dropped."
        self._dataframes[params.get("table", "main")] = df
        return None

    def _act_cast_type(self, params: Dict[str, Any]) -> Optional[str]:
        df = self._get_df(params)
        col = params.get("column")
        target = params.get("target_dtype")
        if col is None or target is None:
            return "Missing required params 'column' and 'target_dtype'."
        if col not in df.columns:
            return f"Column '{col}' does not exist."

        try:
            if target in ("int", "int64"):
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
            elif target in ("float", "float64"):
                # Strip currency symbols first
                if df[col].dtype == object:
                    df[col] = df[col].astype(str).str.replace(r"[\$,]", "", regex=True)
                df[col] = pd.to_numeric(df[col], errors="coerce")
            elif target == "str":
                df[col] = df[col].astype(str)
            elif target in ("datetime", "date"):
                df[col] = pd.to_datetime(df[col], errors="coerce", infer_datetime_format=True)
            else:
                return f"Unsupported target_dtype '{target}'."
        except Exception as exc:
            return f"Cast failed: {exc}"
        return None

    def _act_normalize(self, params: Dict[str, Any]) -> Optional[str]:
        df = self._get_df(params)
        col = params.get("column")
        method = params.get("method", "trim")
        if col is None:
            return "Missing required param 'column'."
        if col not in df.columns:
            return f"Column '{col}' does not exist."

        if method == "trim":
            df[col] = df[col].astype(str).str.strip()
        elif method == "lower":
            df[col] = df[col].astype(str).str.lower()
        elif method == "upper":
            df[col] = df[col].astype(str).str.upper()
        elif method == "strip_currency":
            df[col] = df[col].astype(str).str.replace(r"[\$,]", "", regex=True)
            df[col] = pd.to_numeric(df[col], errors="coerce")
        elif method == "unify_date":
            df[col] = pd.to_datetime(df[col], errors="coerce", infer_datetime_format=True)
            df[col] = df[col].dt.strftime("%Y-%m-%d")
        elif method == "strip_prefix":
            prefix = params.get("prefix", "")
            df[col] = df[col].astype(str).str.replace(prefix, "", regex=False)
        elif method == "map_values":
            mapping = params.get("mapping", {})
            df[col] = df[col].replace(mapping)
        elif method == "clip":
            lower = params.get("lower")
            upper = params.get("upper")
            if df[col].dtype.kind in ("i", "f"):
                df[col] = df[col].clip(lower=lower, upper=upper)
        else:
            return f"Unknown normalize method '{method}'."
        return None

    def _act_join(self, params: Dict[str, Any]) -> Optional[str]:
        right_table = params.get("right_table")
        left_on = params.get("left_on")
        right_on = params.get("right_on")
        how = params.get("how", "left")
        if not all([right_table, left_on, right_on]):
            return "Missing required params: right_table, left_on, right_on."
        if right_table not in self._dataframes:
            return f"Table '{right_table}' not found."
        left_df = self._dataframes.get("main")
        right_df = self._dataframes[right_table]
        if left_df is None:
            return "No 'main' table to join into."
        if left_on not in left_df.columns:
            return f"Column '{left_on}' not in main table."
        if right_on not in right_df.columns:
            return f"Column '{right_on}' not in {right_table}."

        # Type-safe join: coerce both keys to string
        left_df = left_df.copy()
        right_df = right_df.copy()
        left_df[left_on] = left_df[left_on].astype(str)
        right_df[right_on] = right_df[right_on].astype(str)

        try:
            merged = left_df.merge(right_df, left_on=left_on, right_on=right_on, how=how, suffixes=("", "_master"))
            if len(merged) == 0:
                return "Join produced empty result."
            self._dataframes["main"] = merged
        except Exception as exc:
            return f"Join failed: {exc}"
        return None

    def _act_validate(self, _params: Dict[str, Any]) -> Optional[str]:
        """No-op action that just returns current validation errors in the observation."""
        return None

    # ------------------------------------------------------------------
    # Reward computation
    # ------------------------------------------------------------------

    def _compute_reward(self, error_msg: Optional[str]) -> Reward:
        assert self._task is not None
        if error_msg and "destructive" in error_msg.lower():
            return Reward(scalar=normalize_score(0.01), components={"penalty": -1.0}, reasoning=error_msg)

        main_df = self._dataframes.get("main", pd.DataFrame())
        gt_df = self._task.ground_truth_dataframes.get("main", pd.DataFrame())

        step_penalty = 0.01
        
        raw: float = 0.01
        if hasattr(self._task, "grader") and self._task.grader:
            raw = self._task.grader(main_df, gt_df, self._task.business_rules, self._task.target_schema)

        raw = raw - step_penalty
        scalar = normalize_score(raw)

        delta = round(scalar - self._prev_score, 4)
        self._prev_score = scalar

        components = {
            "grader_score": round(raw + step_penalty, 4),
            "step_penalty": step_penalty,
        }
        reasoning = error_msg if error_msg else f"Reward={scalar:.4f}, delta={delta:.4f}"
        return Reward(scalar=scalar, components=components, reasoning=reasoning)

    # ------------------------------------------------------------------
    # Observation builder
    # ------------------------------------------------------------------

    def _build_observation(self) -> Observation:
        assert self._task is not None
        main_df = self._dataframes.get("main", pd.DataFrame())

        # Preview — top 5 rows
        preview = main_df.head(5).fillna("null").to_dict(orient="records")

        # Schema info
        schema_info: Dict[str, ColumnStats] = {}
        for col in main_df.columns:
            series = main_df[col]
            stats = ColumnStats(
                dtype=str(series.dtype),
                null_count=int(series.isna().sum()),
                unique_count=int(series.nunique()),
            )
            if series.dtype.kind in ("i", "f"):
                stats.mean = round(float(series.mean()), 4) if not series.isna().all() else None
                mean_ = series.mean()
                std_ = series.std()
                if std_ and std_ > 0:
                    stats.outliers_detected = int(((series - mean_).abs() > 10 * std_).sum())
            schema_info[col] = stats

        # Validation errors
        validation_errors = self._check_validation()

        # Progress
        gt_df = self._task.ground_truth_dataframes.get("main", pd.DataFrame())
        
        progress = 0.01
        if hasattr(self._task, "grader") and self._task.grader:
            progress = self._task.grader(main_df, gt_df, self._task.business_rules, self._task.target_schema)
            
        progress = normalize_score(progress)

        return Observation(
            dataset_preview=preview,
            schema_info=schema_info,
            validation_errors=validation_errors,
            action_history=list(self._action_history),
            current_step=self._current_step,
            max_steps=self._task.max_steps,
            progress_score=progress,
            dataset_size=len(main_df),
            progress_delta=round(progress - self._prev_score, 4),
        )

    def _check_validation(self) -> List[str]:
        """Return a list of current integrity violations."""
        if self._task is None:
            return []
        errors: List[str] = []
        main_df = self._dataframes.get("main", pd.DataFrame())

        # Check nulls
        for col in main_df.columns:
            nc = int(main_df[col].isna().sum())
            if nc > 0:
                errors.append(f"Column '{col}' has {nc} null values.")

        # Check duplicates
        dup_count = int(main_df.duplicated().sum())
        if dup_count > 0:
            errors.append(f"Dataset has {dup_count} duplicate rows.")

        # Check schema
        for col, expected in self._task.target_schema.items():
            if col not in main_df.columns:
                errors.append(f"Missing expected column '{col}'.")
                continue
            kind = main_df[col].dtype.kind
            if expected in ("int",) and kind not in ("i",):
                errors.append(f"Column '{col}' should be int, got {main_df[col].dtype}.")
            elif expected in ("float",) and kind not in ("f", "i"):
                errors.append(f"Column '{col}' should be float, got {main_df[col].dtype}.")

        return errors
