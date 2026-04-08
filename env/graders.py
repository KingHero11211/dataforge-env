"""Deterministic grading for DataForge-Env.

Every grader returns a float strictly in (0.01, 0.99). No randomness.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def _safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def normalize_score(score: float) -> float:
    """Clamp a raw score to strictly (0.01, 0.99) — never 0.0 or 1.0."""
    try:
        v = float(score)
    except (TypeError, ValueError):
        v = 0.01
    if not (v == v):  # NaN guard
        v = 0.01
    return max(0.01, min(v, 0.99))


# ---------------------------------------------------------------------------
# Component scorers  (each returns a RAW float in [0.0, 1.0] before clamping)
# ---------------------------------------------------------------------------

def _score_dtypes(current: pd.DataFrame, ground_truth: pd.DataFrame) -> float:
    """Fraction of columns whose dtype category matches the ground-truth."""
    if ground_truth is None or ground_truth.empty or len(ground_truth.columns) == 0:
        return 0.9
    matches = 0
    total = 0
    for col in ground_truth.columns:
        total += 1
        if col not in current.columns:
            continue
        gt_kind = ground_truth[col].dtype.kind
        cur_kind = current[col].dtype.kind
        if gt_kind == cur_kind:
            matches += 1
        elif gt_kind in ("i", "f") and cur_kind in ("i", "f"):
            matches += 1  # int/float are interchangeable
    return matches / max(total, 1)


def _score_nulls(current: pd.DataFrame) -> float:
    """Fraction of cells that are NOT null."""
    total_cells = max(current.size, 1)
    null_cells = int(current.isna().sum().sum())
    return 1.0 - null_cells / total_cells


def _score_dupes(current: pd.DataFrame) -> float:
    """Fraction of rows that are NOT duplicates."""
    n_rows = max(len(current), 1)
    dup_count = int(current.duplicated().sum())
    return 1.0 - dup_count / n_rows


def _score_business_rules(
    current: pd.DataFrame,
    constraints: List[str],
    target_schema: Optional[Dict[str, str]] = None,
) -> float:
    """Heuristic constraint checker. Returns fraction of rules satisfied."""
    if not constraints:
        return 0.9  # no constraints → near-perfect

    passed = 0
    for rule in constraints:
        rule_lower = rule.lower()

        # --- No null values ---
        if "no null" in rule_lower or "0 null" in rule_lower:
            col = _extract_col(rule, current.columns)
            if col and current[col].isna().sum() == 0:
                passed += 1
            elif col is None and current.isna().sum().sum() == 0:
                passed += 1
            continue

        # --- No duplicate rows ---
        if "no duplicate" in rule_lower or "0 duplicate" in rule_lower:
            if current.duplicated().sum() == 0:
                passed += 1
            continue

        # --- Whitespace trimmed ---
        if "whitespace" in rule_lower or "trim" in rule_lower:
            col = _extract_col(rule, current.columns)
            if col and current[col].dtype == object:
                trimmed = current[col].dropna().str.strip()
                if (trimmed == current[col].dropna()).all():
                    passed += 1
            else:
                passed += 1  # no string col → rule trivially satisfied
            continue

        # --- Numeric check ---
        if "numeric" in rule_lower or "float" in rule_lower:
            col = _extract_col(rule, current.columns)
            if col and current[col].dtype.kind in ("i", "f"):
                passed += 1
            continue

        # --- Date check ---
        if "iso" in rule_lower or "date" in rule_lower:
            col = _extract_col(rule, current.columns)
            if col:
                try:
                    pd.to_datetime(current[col], format="%Y-%m-%d")
                    passed += 1
                except Exception:
                    pass
            continue

        # --- Outlier cap ---
        if "outlier" in rule_lower or "cap" in rule_lower:
            col = _extract_col(rule, current.columns)
            if col and current[col].dtype.kind in ("i", "f"):
                mean_ = current[col].mean()
                std_ = current[col].std()
                if std_ and std_ > 0 and (current[col] <= mean_ + 10 * std_).all():
                    passed += 1
                elif std_ == 0 or (std_ != std_):
                    passed += 1
            continue

        # --- Non-negative ---
        if "non-negative" in rule_lower or ">= 0" in rule_lower or "cannot be negative" in rule_lower:
            col = _extract_col(rule, current.columns)
            if col and current[col].dtype.kind in ("i", "f"):
                if (current[col] >= 0).all():
                    passed += 1
            continue

        # --- inventory_value rule ---
        if "inventory_value" in rule_lower and "stock_level" in rule_lower:
            req = {"stock_level", "unit_price", "inventory_value"}
            if req.issubset(current.columns):
                expected = (current["stock_level"] * current["unit_price"]).round(2)
                actual = current["inventory_value"].round(2)
                if np.allclose(expected.fillna(0), actual.fillna(0), atol=0.05):
                    passed += 1
            continue

        # --- SKU normalised ---
        if "sku" in rule_lower and ("normali" in rule_lower or "strip" in rule_lower):
            if "sku" in current.columns:
                if not current["sku"].astype(str).str.contains("SKU-", na=False).any():
                    passed += 1
            continue

        # --- Category canonical ---
        if "canonical" in rule_lower or "category" in rule_lower:
            if "category" in current.columns:
                bad = {"Veg", "Fruits"}
                vals = set(current["category"].dropna().unique())
                if not (vals & bad):
                    passed += 1
            continue

        # Unknown rules: strict — don't auto-pass

    return passed / max(len(constraints), 1)


def _extract_col(rule: str, columns: pd.Index) -> Optional[str]:
    """Try to find a column name mentioned in the rule text."""
    for col in columns:
        if col.lower() in rule.lower():
            return col
    return None


# ---------------------------------------------------------------------------
# Public component-score functions (kept for backward compat)
# ---------------------------------------------------------------------------

def check_dtypes(current: pd.DataFrame, ground_truth: pd.DataFrame) -> float:
    """Fraction of columns whose dtype category matches; clamped to (0.01,0.99)."""
    return normalize_score(_score_dtypes(current, ground_truth))


def calculate_f1_similarity(current: pd.DataFrame, ground_truth: pd.DataFrame) -> float:
    """Row-level F1 between current and ground-truth; clamped to (0.01,0.99)."""
    if ground_truth is None or ground_truth.empty:
        return normalize_score(0.9)
    common_cols = sorted(set(current.columns) & set(ground_truth.columns))
    if not common_cols:
        return normalize_score(0.05)

    def _hashes(df: pd.DataFrame) -> set:
        sub = df[common_cols].fillna("__NULL__").astype(str)
        return set(sub.apply(lambda r: "|".join(r), axis=1))

    cur_hashes = _hashes(current)
    gt_hashes = _hashes(ground_truth)

    if not gt_hashes:
        return normalize_score(0.9)

    tp = len(cur_hashes & gt_hashes)
    fp = len(cur_hashes - gt_hashes)
    fn = len(gt_hashes - cur_hashes)

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    if precision + recall == 0:
        return normalize_score(0.05)

    f1 = 2 * precision * recall / (precision + recall)
    return normalize_score(f1)


def verify_business_rules(
    current: pd.DataFrame,
    constraints: List[str],
    target_schema: Optional[Dict[str, str]] = None,
) -> float:
    """Heuristic rule checker; clamped to (0.01,0.99)."""
    return normalize_score(_score_business_rules(current, constraints, target_schema))


# ---------------------------------------------------------------------------
# Internal unified grader
# ---------------------------------------------------------------------------

def _unified_grade(
    current_df: pd.DataFrame,
    ground_truth_df: pd.DataFrame,
    constraints: List[str],
    target_schema: Optional[Dict[str, str]] = None,
) -> float:
    """Combined weighted formula — all sub-scores are raw [0,1] before clamping."""
    c_schema = _score_dtypes(current_df, ground_truth_df)
    c_nulls  = _score_nulls(current_df)
    c_dupes  = _score_dupes(current_df)
    c_logic  = _score_business_rules(current_df, constraints, target_schema)

    raw = 0.3 * c_schema + 0.2 * c_nulls + 0.1 * c_dupes + 0.4 * c_logic
    return normalize_score(raw)


# ---------------------------------------------------------------------------
# Public graders — named EXACTLY as listed in openenv.yaml
# ---------------------------------------------------------------------------

def grade_easy(
    current_df: pd.DataFrame,
    ground_truth_df: pd.DataFrame,
    constraints: List[str],
    target_schema: Optional[Dict[str, str]] = None,
) -> float:
    """Grade the easy task (The Untidy Retailer). Returns float in (0.01, 0.99)."""
    try:
        return _unified_grade(current_df, ground_truth_df, constraints, target_schema)
    except Exception:
        return normalize_score(0.1)


def grade_medium(
    current_df: pd.DataFrame,
    ground_truth_df: pd.DataFrame,
    constraints: List[str],
    target_schema: Optional[Dict[str, str]] = None,
) -> float:
    """Grade the medium task (Financial Anomaly). Returns float in (0.01, 0.99)."""
    try:
        return _unified_grade(current_df, ground_truth_df, constraints, target_schema)
    except Exception:
        return normalize_score(0.1)


def grade_hard(
    current_df: pd.DataFrame,
    ground_truth_df: pd.DataFrame,
    constraints: List[str],
    target_schema: Optional[Dict[str, str]] = None,
) -> float:
    """Grade the hard task (Supply Chain Reconciliation). Returns float in (0.01, 0.99)."""
    try:
        data_score = float(calculate_f1_similarity(current_df, ground_truth_df))
        c_schema   = _score_dtypes(current_df, ground_truth_df)
        c_logic    = _score_business_rules(current_df, constraints, target_schema)
        raw = 0.2 * c_schema + 0.5 * data_score + 0.3 * c_logic
        return normalize_score(raw)
    except Exception:
        return normalize_score(0.1)
