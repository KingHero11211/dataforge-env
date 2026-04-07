#!/usr/bin/env python3
"""inference.py — OpenEnv-compliant inference script for DataForge-Env.

Follows the STRICT [START] / [STEP] / [END] format.

Environment variables required:
  API_BASE_URL  — LLM API base URL
  MODEL_NAME    — model identifier
  HF_TOKEN      — HuggingFace token (used as bearer for the LLM API)
"""

from __future__ import annotations

import json
import os
import sys
import traceback

import requests
from openai import OpenAI

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_BASE_URL: str = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME: str = os.environ.get("MODEL_NAME", "gpt-4o")
HF_TOKEN: str = os.environ.get("HF_TOKEN", "")
ENV_URL: str = os.environ.get("ENV_URL", "http://localhost:7860")

TASK_ID: str = os.environ.get("TASK_ID", "easy")

client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN,
)

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a data-cleaning agent for the DataForge-Env environment.

You receive an observation containing:
- dataset_preview (top 5 rows)
- schema_info (column stats)
- validation_errors (current issues)
- action_history (what you've done)
- progress_score (current cleanliness 0-1)

You must respond with a single JSON action:
{
  "action_type": "<fill_missing|drop_duplicates|cast_type|normalize|join|validate>",
  "params": { ... }
}

Available action types and their params:
- fill_missing: column (str), strategy (mean|median|mode|constant|drop), fill_value (optional)
- drop_duplicates: subset (list[str], optional)
- cast_type: column (str), target_dtype (int|float|str|datetime)
- normalize: column (str), method (trim|lower|upper|strip_currency|unify_date|strip_prefix|map_values|clip)
  - strip_prefix also accepts: prefix (str)
  - map_values also accepts: mapping (dict)
  - clip also accepts: lower, upper
- join: right_table (str), left_on (str), right_on (str), how (left|inner|right|outer)
- validate: {} (no params)

Respond ONLY with the JSON. NO explanation text.
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def env_reset(task_id: str) -> dict:
    resp = requests.post(f"{ENV_URL}/reset", json={"task_id": task_id}, timeout=30)
    resp.raise_for_status()
    return resp.json()


def env_step(action: dict) -> dict:
    resp = requests.post(f"{ENV_URL}/step", json={"action": action}, timeout=30)
    resp.raise_for_status()
    return resp.json()


def ask_llm(observation: dict) -> dict:
    """Query the LLM for the next action given the current observation."""
    user_msg = json.dumps(observation, indent=2, default=str)
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.0,
        max_tokens=512,
    )
    content = response.choices[0].message.content.strip()
    # Strip markdown fences if present
    if content.startswith("```"):
        content = content.split("\n", 1)[1] if "\n" in content else content[3:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()
    return json.loads(content)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def log_start(task, env, model):
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step, action, reward, done, error):
    error_val = error if error else "null"
    print(
        f"[STEP] step={step} action={json.dumps(action)} "
        f"reward={reward:.2f} done={'true' if done else 'false'} error={error_val}",
        flush=True
    )

def log_end(success, steps, rewards):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={'true' if success else 'false'} steps={steps} rewards={rewards_str}",
        flush=True
    )


def main():
    task_id = TASK_ID
    rewards = []

    reset_data = env_reset(task_id)
    observation = reset_data["observation"]

    done = False
    step_num = 0

    log_start(task_id, "dataforge_env", MODEL_NAME)

    while not done:
        step_num += 1

        try:
            action = ask_llm(observation)
        except Exception:
            action = {"action_type": "validate", "params": {}}

        step_data = env_step(action)
        observation = step_data["observation"]
        reward_val = step_data["reward"]["scalar"]
        done = step_data["done"]
        error = step_data.get("info", {}).get("error")

        rewards.append(reward_val)

        log_step(step_num, action, reward_val, done, error)

        if done:
            break

    total_score = sum(rewards) / len(rewards) if rewards else 0.001
    total_score = min(max(total_score, 0.001), 0.999)

    success = total_score >= 0.7

    log_end(success, step_num, rewards)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        # ALWAYS print END even on failure — validator requires it
        log_end(False, 0, [])
        sys.exit(1)
