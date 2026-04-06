---
title: DataForge Env
emoji: 🤖
colorFrom: blue
colorTo: green
sdk: docker
app_file: server/app.py
pinned: false
---

# DataForge-Env 🔧

> A production-grade OpenEnv environment for evaluating LLM-based agents on real-world data cleaning, validation, and multi-table reconciliation tasks.

---

## Overview

DataForge-Env wraps a stateful data-cleaning sandbox. Agents interact via structured actions to fix dirty datasets — filling nulls, casting types, normalising values, joining tables, and satisfying business rules.

Rewards are **dense** and **deterministic**: agents receive granular feedback after every step, enabling RL-style training and scientific benchmarking.

## Tasks

| ID | Name | Difficulty | Max Steps | Description |
|----|------|-----------|-----------|-------------|
| `easy` | The Untidy Retailer | Easy | 15 | Fill missing emails, remove duplicates, trim whitespace |
| `medium` | Financial Anomaly | Medium | 20 | Parse currency strings, unify dates, cap outliers |
| `hard` | Supply Chain Reconciliation | Hard | 25 | Normalise SKU keys, join tables, compute inventory value |

## Quick Start

### Local
```bash
pip install -r requirements.txt
python -m server.app
# Server runs on http://localhost:7860
```

### Docker
```bash
docker build -t dataforge-env .
docker run -p 7860:7860 dataforge-env
```

### API Usage

**Reset** (start an episode):
```bash
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "easy"}'
```

**Step** (apply an action):
```bash
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"action": {"action_type": "fill_missing", "params": {"column": "email", "strategy": "constant", "fill_value": "unknown@example.com"}}}'
```

## Action Space

| Action | Key Params |
|--------|-----------|
| `fill_missing` | `column`, `strategy` (mean/median/mode/constant/drop), `fill_value` |
| `drop_duplicates` | `subset` (optional list of columns) |
| `cast_type` | `column`, `target_dtype` (int/float/str/datetime) |
| `normalize` | `column`, `method` (trim/lower/upper/strip_currency/unify_date/strip_prefix/map_values/clip) |
| `join` | `right_table`, `left_on`, `right_on`, `how` |
| `validate` | (no params — returns current validation errors) |

## Reward Formula

```
R = 0.3 × C_schema + 0.2 × C_nulls + 0.1 × C_dupes + 0.4 × C_logic − 0.01 × step_penalty
```

All components and the final reward are normalised to **[0, 1]**.

## Inference Script

```bash
export API_BASE_URL=https://api-inference.huggingface.co/v1
export MODEL_NAME=meta-llama/Llama-3-70B-Instruct
export HF_TOKEN=hf_...
export ENV_URL=http://localhost:7860
export TASK_ID=easy

python inference.py
```

Output follows strict `[START]` / `[STEP]` / `[END]` format.

## Project Structure

```
├── openenv.yaml          # Environment specification
├── env/
│   ├── models.py         # Pydantic schemas
│   ├── env.py            # Core environment class
│   ├── tasks.py          # Task definitions & data generators
│   └── graders.py        # Deterministic grading
├── server/
│   └── app.py            # FastAPI server
├── inference.py          # LLM agent inference script
├── Dockerfile            # Container definition
├── requirements.txt      # Python dependencies
└── README.md             # This file
```

## License

MIT
