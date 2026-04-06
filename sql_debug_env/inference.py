"""
inference.py - Baseline inference script for SQL Debug Env
Mandatory stdout format: [START], [STEP], [END] lines per the hackathon spec.
"""

import asyncio
import os
import re
import textwrap
from typing import List, Optional

from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env file
load_dotenv()

# Import env client from the package
try:
    from sql_debug_env import SqlDebugAction, SqlDebugEnv
except ImportError:
    # fallback for running directly from project root (python inference.py)
    from client import SqlDebugEnv  # type: ignore
    from models import SqlDebugAction  # type: ignore

# Mandatory env vars (loaded from .env file via load_dotenv())
API_BASE_URL: str = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY: str = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY", "")
MODEL_NAME: str = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-Coder-32B-Instruct")
LOCAL_IMAGE_NAME: str = os.getenv("LOCAL_IMAGE_NAME", "sql_debug_env:latest")

# Episode config
BENCHMARK = "sql_debug_env"
MAX_STEPS = 5
TEMPERATURE = 0.1
MAX_TOKENS = 512
SUCCESS_THRESHOLD = 0.5  # score >= this = success
TASK_IDS = ["fix_query", "write_join", "optimize_query"]


# Mandatory stdout log functions
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    # action must be a single line, no newlines
    action_clean = action.replace("\n", " ").replace("\r", " ")[:200]
    error_val = error.replace("\n", " ")[:100] if error else "null"
    print(
        f"[STEP] step={step} action={action_clean} "
        f"reward={reward:.2f} done={str(done).lower()} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


# SQL extractor
def extract_sql(text: str) -> str:
    """Extract SQL from model response. Looks for ```sql blocks first."""
    m = re.search(r"```sql\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    m = re.search(r"```\s*(SELECT.*?)```", text, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    m = re.search(r"(SELECT\s+.+?;)", text, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return text.strip()


# Prompts
SYSTEM_PROMPT = textwrap.dedent(
    """
    You are an expert SQL developer. You write correct, efficient SQLite queries.

    Rules:
    1. Output your SQL inside a ```sql code block - nothing else outside it.
    2. Use only SQLite-compatible syntax.
    3. If fixing a broken query, find ALL bugs and fix them in one shot.
    4. If writing from scratch, read the schema carefully first.
    5. Do not write explanations outside the code block.

    Example correct output:
    ```sql
    SELECT name, email FROM customers WHERE city = 'Mumbai' ORDER BY name ASC;
    ```
"""
).strip()


def build_prompt(obs_data: dict, step: int, prev_error: str = "") -> str:
    parts = [
        f"Step {step}/{MAX_STEPS}",
        f"\nTask:\n{obs_data.get('task_description', '')}",
        f"\nSchema:\n{obs_data.get('schema_info', '')}",
    ]
    if obs_data.get("broken_query"):
        parts.append(f"\nBroken query to fix or optimize:\n{obs_data['broken_query']}")
    if obs_data.get("explain_plan"):
        parts.append(f"\nEXPLAIN QUERY PLAN (shows the problem):\n{obs_data['explain_plan']}")
    if prev_error:
        parts.append(f"\nYour previous query caused this error: {prev_error}")
        parts.append("Fix it and try again.")
    parts.append("\nWrite the correct SQL query now:")
    return "\n".join(parts)


# LLM call
def call_llm(client: OpenAI, prompt: str) -> str:
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        print(f"[DEBUG] LLM call failed: {e}", flush=True)
        return ""


# Run one task episode
async def run_task(env: SqlDebugEnv, client: OpenAI, task_id: str) -> float:
    """Run one full episode for a task. Returns final score in [0, 1]."""
    rewards: List[float] = []
    steps_taken = 0
    best_score = 0.0
    success = False
    prev_error = ""

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        result = await env.reset(task_id=task_id)
        obs = result.observation

        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            obs_dict = {
                "task_description": obs.task_description,
                "schema_info": obs.schema_info,
                "broken_query": obs.broken_query,
                "explain_plan": obs.explain_plan,
            }
            prompt = build_prompt(obs_dict, step, prev_error)
            response = call_llm(client, prompt)
            sql = extract_sql(response) or "SELECT 1;"

            result = await env.step(SqlDebugAction(sql_query=sql))
            obs = result.observation
            reward = result.reward or 0.0
            done = result.done
            error = obs.error_message

            rewards.append(reward)
            steps_taken = step
            prev_error = error or ""

            if reward > best_score:
                best_score = reward

            log_step(step=step, action=sql, reward=reward, done=done, error=error)

            if done:
                break

        score = round(best_score, 2)
        success = score >= SUCCESS_THRESHOLD

    except Exception as e:
        print(f"[DEBUG] Task {task_id} crashed: {e}", flush=True)
        score = 0.0

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return score


# Main: run all 3 tasks
async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    all_scores = {}

    for task_id in TASK_IDS:
        # Each task gets a fresh env connection (fresh Docker container state)
        env = await SqlDebugEnv.from_docker_image(LOCAL_IMAGE_NAME)
        try:
            score = await run_task(env, client, task_id)
            all_scores[task_id] = score
        finally:
            await env.close()

    print("\n[SUMMARY]", flush=True)
    for tid, sc in all_scores.items():
        print(f"  {tid}: {sc:.2f}", flush=True)
    avg = sum(all_scores.values()) / len(all_scores)
    print(f"  average: {avg:.2f}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
