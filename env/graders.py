"""Deterministic grading for DataForge-Env.

Every grader returns a float in [0, 1].  No randomness.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


# ---------------------------------------------------------------------------
# Component scores
# ---------------------------------------------------------------------------

def check_dtypes(current: pd.DataFrame, ground_truth: pd.DataFrame) -> float:
    """Fraction of columns whose dtype category matches the ground-truth."""
    if ground_truth.empty:
        return 1.0
    matches = 0
    total = 0
    for col in ground_truth.columns:
        if col not in current.columns:
            total += 1
            continue
        total += 1
        gt_kind = ground_truth[col].dtype.kind
        cur_kind = current[col].dtype.kind
        if gt_kind == cur_kind:
            matches += 1
        elif gt_kind in ("i", "f") and cur_kind in ("i", "f"):
            matches += 1  # int/float are close enough
    return matches / max(total, 1)


def calculate_f1_similarity(current: pd.DataFrame, ground_truth: pd.DataFrame) -> float:
    """Row-level F1 between current and ground-truth (based on string hashing)."""
    if ground_truth.empty:
        return 1.0

    common_cols = sorted(set(current.columns) & set(ground_truth.columns))
    if not common_cols:
        return 0.0

    def _row_hashes(df: pd.DataFrame) -> set:
        sub = df[common_cols].fillna("__NULL__").astype(str)
        return set(sub.apply(lambda r: "|".join(r), axis=1))

    cur_hashes = _row_hashes(current)
    gt_hashes = _row_hashes(ground_truth)

    if not gt_hashes:
        return 1.0

    tp = len(cur_hashes & gt_hashes)
    fp = len(cur_hashes - gt_hashes)
    fn = len(gt_hashes - cur_hashes)

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def verify_business_rules(
    current: pd.DataFrame,
    constraints: List[str],
    target_schema: Dict[str, str] | None = None,
) -> float:
    """Heuristic constraint checker.  Returns fraction of rules satisfied."""
    if not constraints:
        return 1.0

    passed = 0
    for rule in constraints:
        rule_lower = rule.lower()

        # --- No null values ---
        if "no null" in rule_lower or "0 null" in rule_lower:
            col = _extract_column_from_rule(rule, current.columns)
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
            col = _extract_column_from_rule(rule, current.columns)
            if col and current[col].dtype == object:
                trimmed = current[col].dropna().str.strip()
                if (trimmed == current[col].dropna()).all():
                    passed += 1
            else:
                passed += 1
            continue

        # --- Numeric check ---
        if "numeric" in rule_lower or "float" in rule_lower:
            col = _extract_column_from_rule(rule, current.columns)
            if col and current[col].dtype.kind in ("i", "f"):
                passed += 1
            continue

        # --- Date check ---
        if "iso" in rule_lower or "date" in rule_lower:
            col = _extract_column_from_rule(rule, current.columns)
            if col:
                try:
                    pd.to_datetime(current[col], format="%Y-%m-%d")
                    passed += 1
                except Exception:
                    pass
            continue

        # --- Outlier cap ---
        if "outlier" in rule_lower or "cap" in rule_lower:
            col = _extract_column_from_rule(rule, current.columns)
            if col and current[col].dtype.kind in ("i", "f"):
                mean_ = current[col].mean()
                std_ = current[col].std()
                if std_ > 0 and (current[col] <= mean_ + 10 * std_).all():
                    passed += 1
                elif std_ == 0:
                    passed += 1
            continue

        # --- Non-negative ---
        if "non-negative" in rule_lower or ">= 0" in rule_lower or "cannot be negative" in rule_lower:
            col = _extract_column_from_rule(rule, current.columns)
            if col and current[col].dtype.kind in ("i", "f"):
                if (current[col] >= 0).all():
                    passed += 1
            continue

        # --- inventory_value rule ---
        if "inventory_value" in rule_lower and "stock_level" in rule_lower:
            if {"stock_level", "unit_price", "inventory_value"}.issubset(current.columns):
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
                if not vals & bad:
                    passed += 1
            continue

        # Fallback: count as passed if we can't parse
        passed += 0  # strict — unknown rules don't auto-pass

    return passed / max(len(constraints), 1)


def _extract_column_from_rule(rule: str, columns: pd.Index) -> str | None:
    """Try to find a column name mentioned in the rule text."""
    for col in columns:
        if col.lower() in rule.lower():
            return col
    return None


# ---------------------------------------------------------------------------
# Public grading API
# ---------------------------------------------------------------------------

def grade_task(
    current_df: pd.DataFrame,
    ground_truth_df: pd.DataFrame,
    constraints: List[str],
    target_schema: Dict[str, str] | None = None,
) -> float:
    """Return a deterministic score in [0, 1]."""
    schema_score = check_dtypes(current_df, ground_truth_df)
    data_score = calculate_f1_similarity(current_df, ground_truth_df)
    constraint_score = verify_business_rules(current_df, constraints, target_schema)

    final = 0.2 * schema_score + 0.5 * data_score + 0.3 * constraint_score
    return round(min(max(final, 0.0), 1.0), 4)
