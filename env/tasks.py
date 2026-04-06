"""Deterministic task definitions and synthetic dataset generators for DataForge-Env."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Task descriptor
# ---------------------------------------------------------------------------

@dataclass
class TaskDefinition:
    task_id: str
    name: str
    difficulty: str
    max_steps: int
    description: str
    # Populated at runtime
    dirty_dataframes: Dict[str, pd.DataFrame] = field(default_factory=dict)
    ground_truth_dataframes: Dict[str, pd.DataFrame] = field(default_factory=dict)
    target_schema: Dict[str, str] = field(default_factory=dict)
    business_rules: List[str] = field(default_factory=list)


# ===================================================================
# EASY — "The Untidy Retailer"
# ===================================================================

def _build_easy_task() -> TaskDefinition:
    """Generate a deterministic 1 000-row customers dataset with known dirt."""
    rng = np.random.RandomState(42)

    n = 1000
    first_names = ["Alice", "Bob", "Charlie", "Diana", "Eve", "Frank",
                   "Grace", "Hank", "Ivy", "Jack"]
    last_names = ["Smith", "Johnson", "Williams", "Brown", "Jones",
                  "Garcia", "Miller", "Davis", "Rodriguez", "Martinez"]

    names = [
        f"  {first_names[rng.randint(0, len(first_names))]} {last_names[rng.randint(0, len(last_names))]}  "
        for _ in range(n)
    ]
    emails = [f"user{i}@example.com" for i in range(n)]
    ages = rng.randint(18, 80, size=n).tolist()
    cities = rng.choice(["New York", "Los Angeles", "Chicago", "Houston", "Phoenix"], size=n).tolist()

    df = pd.DataFrame({"name": names, "email": emails, "age": ages, "city": cities})

    # Inject 15 % missing emails (deterministic indices)
    null_indices = rng.choice(n, size=int(n * 0.15), replace=False)
    df.loc[null_indices, "email"] = None

    # Inject 5 % duplicate rows
    dup_indices = rng.choice(n, size=int(n * 0.05), replace=False)
    dups = df.iloc[dup_indices].copy()
    df = pd.concat([df, dups], ignore_index=True)

    # --- Ground truth ---
    gt = df.copy()
    gt["name"] = gt["name"].str.strip()
    gt["email"] = gt["email"].fillna("unknown@example.com")
    gt = gt.drop_duplicates().reset_index(drop=True)

    target_schema = {"name": "str", "email": "str", "age": "int", "city": "str"}
    rules = [
        "No null values in email column",
        "No duplicate rows",
        "name column must have no leading/trailing whitespace",
    ]

    task = TaskDefinition(
        task_id="easy",
        name="The Untidy Retailer",
        difficulty="easy",
        max_steps=15,
        description="Clean a customers dataset: fill missing emails, remove duplicates, trim names.",
        target_schema=target_schema,
        business_rules=rules,
    )
    task.dirty_dataframes["main"] = df
    task.ground_truth_dataframes["main"] = gt
    return task


# ===================================================================
# MEDIUM — "Financial Anomaly"
# ===================================================================

def _build_medium_task() -> TaskDefinition:
    rng = np.random.RandomState(123)
    n = 800

    amounts_raw: List[str] = []
    amounts_clean: List[float] = []
    for _ in range(n):
        val = round(rng.uniform(5.0, 5000.0), 2)
        amounts_clean.append(val)
        amounts_raw.append(f"${val:,.2f}")

    dates_raw: List[str] = []
    dates_clean: List[str] = []
    base_year = 2024
    for i in range(n):
        m = rng.randint(1, 13)
        d = rng.randint(1, 29)
        iso = f"{base_year}-{m:02d}-{d:02d}"
        dates_clean.append(iso)
        if i % 3 == 0:
            dates_raw.append(f"{m:02d}/{d:02d}/{base_year}")  # US format
        else:
            dates_raw.append(iso)

    categories = rng.choice(["Food", "Electronics", "Clothing", "Travel"], size=n).tolist()

    df = pd.DataFrame({
        "transaction_id": list(range(1, n + 1)),
        "amount": amounts_raw,
        "date": dates_raw,
        "category": categories,
    })

    # Inject outliers (2 %)
    outlier_idx = rng.choice(n, size=int(n * 0.02), replace=False)
    clean_amounts = list(amounts_clean)
    mean_a = float(np.mean(amounts_clean))
    std_a = float(np.std(amounts_clean))
    cap = round(mean_a + 10 * std_a, 2)
    for idx in outlier_idx:
        extreme = round(mean_a + 15 * std_a, 2)
        df.at[idx, "amount"] = f"${extreme:,.2f}"
        clean_amounts[idx] = cap  # capped

    gt = pd.DataFrame({
        "transaction_id": list(range(1, n + 1)),
        "amount": clean_amounts,
        "date": dates_clean,
        "category": categories,
    })
    gt["amount"] = gt["amount"].astype(float)
    gt["date"] = pd.to_datetime(gt["date"]).dt.strftime("%Y-%m-%d")

    target_schema = {
        "transaction_id": "int",
        "amount": "float",
        "date": "datetime",
        "category": "str",
    }
    rules = [
        "amount must be numeric float",
        "date must be ISO-8601 string (YYYY-MM-DD)",
        "outlier amounts (>10 std) must be capped",
    ]

    task = TaskDefinition(
        task_id="medium",
        name="Financial Anomaly",
        difficulty="medium",
        max_steps=20,
        description="Clean transactions: parse currency strings, unify dates, cap outliers.",
        target_schema=target_schema,
        business_rules=rules,
    )
    task.dirty_dataframes["main"] = df
    task.ground_truth_dataframes["main"] = gt
    return task


# ===================================================================
# HARD — "Supply Chain Reconciliation"
# ===================================================================

def _build_hard_task() -> TaskDefinition:
    rng = np.random.RandomState(777)
    n_logs = 500
    n_master = 50

    # --- Warehouse master ---
    skus_master = [str(i) for i in range(100, 100 + n_master)]
    prices = [round(rng.uniform(1.0, 200.0), 2) for _ in range(n_master)]
    cats_raw = rng.choice(["Veg", "Vegetables", "Fruit", "Fruits", "Dairy", "Beverages"], size=n_master).tolist()
    cats_clean = []
    mapping = {"Veg": "Vegetables", "Fruits": "Fruit"}
    for c in cats_raw:
        cats_clean.append(mapping.get(c, c))

    master = pd.DataFrame({
        "sku": skus_master,
        "product_name": [f"Product_{s}" for s in skus_master],
        "unit_price": prices,
        "category": cats_raw,
    })

    # --- Inventory logs ---
    log_skus_raw = [f"SKU-{rng.choice(skus_master)}" for _ in range(n_logs)]
    stock_levels = rng.randint(-5, 200, size=n_logs).tolist()
    dates_log = [f"2024-{rng.randint(1,13):02d}-{rng.randint(1,29):02d}" for _ in range(n_logs)]

    logs = pd.DataFrame({
        "log_id": list(range(1, n_logs + 1)),
        "sku": log_skus_raw,
        "stock_level": stock_levels,
        "log_date": dates_log,
    })

    # --- Ground truth ---
    gt_logs = logs.copy()
    gt_logs["sku_clean"] = gt_logs["sku"].str.replace("SKU-", "", regex=False)
    gt_master = master.copy()
    gt_master["category"] = [mapping.get(c, c) for c in gt_master["category"]]

    gt_merged = gt_logs.merge(gt_master, left_on="sku_clean", right_on="sku", how="left", suffixes=("", "_master"))
    gt_merged["stock_level"] = gt_merged["stock_level"].clip(lower=0)
    gt_merged["inventory_value"] = (gt_merged["stock_level"] * gt_merged["unit_price"]).round(2)
    gt_merged["inventory_value"] = gt_merged["inventory_value"].fillna(0.0)

    keep_cols = [
        "log_id", "sku_clean", "stock_level", "log_date",
        "product_name", "unit_price", "category", "inventory_value",
    ]
    gt_merged = gt_merged[[c for c in keep_cols if c in gt_merged.columns]].copy()
    gt_merged = gt_merged.rename(columns={"sku_clean": "sku"})

    target_schema = {
        "log_id": "int",
        "sku": "str",
        "stock_level": "int",
        "log_date": "str",
        "product_name": "str",
        "unit_price": "float",
        "category": "str",
        "inventory_value": "float",
    }
    rules = [
        "SKU keys must be normalised (strip 'SKU-' prefix)",
        "Category names must be canonical (Veg→Vegetables, Fruits→Fruit)",
        "stock_level must be >= 0",
        "inventory_value = stock_level * unit_price",
    ]

    task = TaskDefinition(
        task_id="hard",
        name="Supply Chain Reconciliation",
        difficulty="hard",
        max_steps=25,
        description="Reconcile inventory logs and warehouse master: normalise keys, join, clamp stock, compute value.",
        target_schema=target_schema,
        business_rules=rules,
    )
    task.dirty_dataframes["main"] = logs
    task.dirty_dataframes["warehouse_master"] = master
    task.ground_truth_dataframes["main"] = gt_merged
    return task


# ---------------------------------------------------------------------------
# Public registry
# ---------------------------------------------------------------------------

TASK_REGISTRY: Dict[str, Any] = {
    "easy": _build_easy_task,
    "medium": _build_medium_task,
    "hard": _build_hard_task,
}


def get_task(task_id: str) -> TaskDefinition:
    """Return a freshly-built TaskDefinition (deterministic, no shared state)."""
    builder = TASK_REGISTRY.get(task_id)
    if builder is None:
        raise ValueError(f"Unknown task_id '{task_id}'. Choose from: {list(TASK_REGISTRY.keys())}")
    return builder()
