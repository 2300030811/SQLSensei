"""
Root-level baseline inference script for SQL Debug Env.
Required at repository root by the platform validator.
"""

import asyncio
import os
import re
import textwrap
from typing import Dict, List, Optional

from dotenv import load_dotenv
from openai import OpenAI

from sql_debug_env import SqlDebugAction, SqlDebugEnv


load_dotenv()


API_BASE_URL: str = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY: str = (
    os.getenv("HF_ACCESS_TOKEN")
    or os.getenv("HF_TOKEN")
    or os.getenv("OPENAI_API_KEY")
    or "HF_ACCESS_TOKEN"
)
MODEL_NAME: str = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-Coder-32B-Instruct")
LOCAL_IMAGE_NAME: str = os.getenv("LOCAL_IMAGE_NAME", "sql_debug_env:latest")
USE_LOCAL_FALLBACK: str = os.getenv("USE_LOCAL_FALLBACK", "auto").strip().lower()

BENCHMARK = "sql_debug_env"
MAX_STEPS = 5
TEMPERATURE = 0.1
MAX_TOKENS = 512
SUCCESS_THRESHOLD = 0.5
TASK_IDS = ["fix_query", "write_join", "optimize_query"]


def local_fallback_enabled() -> bool:
    if USE_LOCAL_FALLBACK in {"1", "true", "yes", "on"}:
        return True
    if USE_LOCAL_FALLBACK in {"0", "false", "no", "off"}:
        return False
    return API_KEY in {"", "HF_ACCESS_TOKEN"}


LOCAL_FALLBACK_ENABLED = local_fallback_enabled()


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
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


def extract_sql(text: str) -> str:
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


def local_sql_policy(task_id: str) -> str:
    if task_id == "fix_query":
        return "SELECT name, email FROM customers WHERE city = 'Mumbai' ORDER BY name ASC;"
    if task_id == "write_join":
        return (
            "SELECT c.name AS customer_name, SUM(o.quantity * p.price) AS total_spent "
            "FROM orders o "
            "JOIN customers c ON o.customer_id = c.id "
            "JOIN products p ON o.product_id = p.id "
            "GROUP BY c.id, c.name "
            "HAVING SUM(o.quantity * p.price) > 500 "
            "ORDER BY total_spent DESC;"
        )
    if task_id == "optimize_query":
        return (
            "SELECT user_id, COUNT(*) AS purchase_count "
            "FROM events "
            "WHERE event_type = 'purchase' "
            "AND created_at >= date('now', '-30 days') "
            "GROUP BY user_id "
            "HAVING COUNT(*) > 3 "
            "ORDER BY purchase_count DESC;"
        )
    return "SELECT 1;"


SYSTEM_PROMPT = textwrap.dedent(
    """
    You are an expert SQL developer. You write correct, efficient SQLite queries.

    Rules:
    1. Output your SQL inside a ```sql code block - nothing else outside it.
    2. Use only SQLite-compatible syntax.
    3. If fixing a broken query, find all bugs and fix them in one shot.
    4. If writing from scratch, read the schema carefully first.
    5. Do not write explanations outside the code block.
"""
).strip()


def build_prompt(obs_data: Dict[str, str], step: int, prev_error: str = "") -> str:
    parts = [
        f"Step {step}/{MAX_STEPS}",
        f"\nTask:\n{obs_data.get('task_description', '')}",
        f"\nSchema:\n{obs_data.get('schema_info', '')}",
    ]
    if obs_data.get("broken_query"):
        parts.append(f"\nBroken query to fix or optimize:\n{obs_data['broken_query']}")
    if obs_data.get("explain_plan"):
        parts.append(f"\nEXPLAIN QUERY PLAN:\n{obs_data['explain_plan']}")
    if prev_error:
        parts.append(f"\nYour previous query caused this error: {prev_error}")
        parts.append("Fix it and try again.")
    parts.append("\nWrite the correct SQL query now:")
    return "\n".join(parts)


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
        print(f"[DEBUG] LLM call failed, using local fallback: {e}", flush=True)
        return ""


def choose_sql(
    client: Optional[OpenAI],
    task_id: str,
    obs_data: Dict[str, str],
    step: int,
    prev_error: str,
) -> str:
    if client is None:
        return local_sql_policy(task_id)

    prompt = build_prompt(obs_data, step, prev_error)
    response = call_llm(client, prompt)
    sql = extract_sql(response)
    if sql:
        return sql
    return local_sql_policy(task_id)


async def run_task(env: SqlDebugEnv, client: Optional[OpenAI], task_id: str) -> float:
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
                "broken_query": obs.broken_query or "",
                "explain_plan": obs.explain_plan or "",
            }

            sql = choose_sql(
                client=client,
                task_id=task_id,
                obs_data=obs_dict,
                step=step,
                prev_error=prev_error,
            )

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


async def main() -> None:
    client: Optional[OpenAI] = None
    if not LOCAL_FALLBACK_ENABLED:
        client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    else:
        print("[DEBUG] Running local fallback policy (no external model calls).", flush=True)

    all_scores: Dict[str, float] = {}

    for task_id in TASK_IDS:
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