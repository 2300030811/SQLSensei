---
title: SQL Debug Env
emoji: 🗄️
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
  - sql
  - code-debugging
  - reinforcement-learning
---

# SQL Debug Env

An OpenEnv reinforcement-learning environment where AI agents learn to debug,
write, and optimize SQL queries against a live SQLite database.

## Why SQL?

SQL fluency is one of the most valuable and verifiable skills for data agents.
Every grader executes the agent's query and checks the actual results — no
fuzzy scoring, no LLM-as-judge. Three tasks test distinct SQL competencies
across an easy → medium → hard progression.

## Tasks

| Task | Difficulty | Description |
|------|-----------|-------------|
| `fix_query` | Easy | Fix a broken SQL query with wrong column names |
| `write_join` | Medium | Write a 3-table JOIN from a natural-language spec |
| `optimize_query` | Hard | Rewrite a slow query to use the correct index |

## Action Space

| Field | Type | Description |
|-------|------|-------------|
| `sql_query` | `str` | The SQL query the agent submits for grading |

## Observation Space

| Field | Type | Description |
|-------|------|-------------|
| `task_id` | `str` | Active task: `fix_query` \| `write_join` \| `optimize_query` |
| `task_description` | `str` | Full natural-language task description |
| `schema_info` | `str` | `CREATE TABLE` DDL for all tables in this episode's DB |
| `broken_query` | `str?` | Provided for tasks 1 and 3 — the query to fix |
| `explain_plan` | `str?` | EXPLAIN QUERY PLAN output (task 3 only) |
| `query_result` | `list?` | Rows returned by the agent's last SQL |
| `error_message` | `str?` | SQL execution error, if any |
| `grading_info` | `dict` | Step-by-step breakdown of how this reward was earned |
| `reward` | `float` | Score for this step, `0.0 – 1.0` |
| `done` | `bool` | `true` when the episode ends |

## Reward Function

All tasks use layered partial credit so partial progress is always rewarded:

**Task 1 — fix_query**
- `+0.20` SQL contains `SELECT` keyword
- `+0.30` Executes without error
- `+0.50` Exact row match (correct order); `+0.25` if unordered match

**Task 2 — write_join**
- `+0.10` Contains `JOIN`
- `+0.10` Contains `GROUP BY`
- `+0.10` Executes without error
- `+0.20` Returns correct column names (`customer_name`, `total_spent`)
- `+0.20` Correct number of result rows
- `+0.30` Exact data match; `+0.15` if unordered match

**Task 3 — optimize_query**
- `+0.10` SQL contains `SELECT`
- `+0.15` Executes without error
- `+0.35` Results match the reference query output
- `+0.40` `EXPLAIN QUERY PLAN` shows index seek on `idx_events_type_date` (not a full scan)

**Step penalty:** −0.05 per step beyond the first (max 5 steps per episode).

## Quick Start

```python
import asyncio
from sql_debug_env import SqlDebugAction, SqlDebugEnv

async def main():
    env = await SqlDebugEnv.from_docker_image("sql-debug-env:latest")
    try:
        result = await env.reset(task_id="fix_query")
        print(result.observation.task_description)

        result = await env.step(SqlDebugAction(
            sql_query="SELECT name, email FROM customers "
                      "WHERE city = 'Mumbai' ORDER BY name ASC"
        ))
        print(f"Reward: {result.reward}")
        print(f"Done: {result.done}")
    finally:
        await env.close()

asyncio.run(main())
```

## Building the Docker Image

```bash
docker build -t sql-debug-env:latest .
docker run -p 8000:8000 sql-debug-env:latest
```

## Deploying to Hugging Face Spaces

```bash
openenv push --repo-id YOUR_HF_USERNAME/sql-debug-env
```

## Running the Baseline Agent

```bash
export API_BASE_URL="https://router.huggingface.co/v1"
export HF_TOKEN="your_token"
export MODEL_NAME="Qwen/Qwen2.5-Coder-32B-Instruct"
export LOCAL_IMAGE_NAME="sql-debug-env:latest"

python inference.py
```

Expected output:
```
[START] task=fix_query env=sql_debug_env model=Qwen/Qwen2.5-Coder-32B-Instruct
[STEP] step=1 action=SELECT name, email FROM customers ... reward=1.00 done=true error=null
[END] success=true steps=1 score=1.00 rewards=1.00
```

## Project Structure

```
sql_debug_env/
├── inference.py              ← Baseline agent (mandatory filename)
├── openenv.yaml              ← OpenEnv manifest
├── pyproject.toml            ← Dependencies
├── __init__.py               ← Package exports
├── client.py                 ← SqlDebugEnv WebSocket client
├── models.py                 ← Action / Observation / State schemas
└── server/
    ├── Dockerfile            ← Container build (uses openenv-base)
    ├── app.py                ← FastAPI + WebSocket server
    ├── sql_debug_env_environment.py  ← Core environment logic
    ├── task_registry.py      ← 3 task definitions with DDL + seed data
    └── graders.py            ← Deterministic scoring functions
```
