"""
SQL Debug Environment server implementation.
Handles 3 SQL tasks with in-memory SQLite databases.
"""

import random
import sqlite3
import uuid
from datetime import datetime, timedelta
from typing import Optional

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import SqlDebugAction, SqlDebugObservation, SqlDebugState
    from .task_registry import TASKS, Task
    from .graders import GRADER_MAP
except ImportError:
    from models import SqlDebugAction, SqlDebugObservation, SqlDebugState
    from server.task_registry import TASKS, Task
    from server.graders import GRADER_MAP

MAX_STEPS = 5


class SqlDebugEnvironment(Environment):
    """
    SQL debugging environment backed by in-memory SQLite.
    One fresh DB per episode. Agent submits SQL, gets scored reward.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        self._state = SqlDebugState(episode_id=str(uuid.uuid4()))
        self._conn: Optional[sqlite3.Connection] = None
        self._current_task: Optional[Task] = None

    def reset(self, seed=None, episode_id=None, task_id: str = "fix_query", **kwargs) -> SqlDebugObservation:
        """Start a new episode. task_id chooses which of the 3 tasks to run."""
        if task_id not in TASKS:
            task_id = "fix_query"

        task = TASKS[task_id]
        self._current_task = task

        # Fresh in-memory SQLite DB for every episode
        if self._conn:
            try:
                self._conn.close()
            except Exception:
                pass
        self._conn = sqlite3.connect(":memory:")

        # Create tables
        for stmt in task.ddl.strip().split(";"):
            s = stmt.strip()
            if s:
                self._conn.execute(s)
        self._conn.commit()

        # Seed data
        if task.task_id == "optimize_query":
            self._seed_events(self._conn)
        else:
            for stmt in task.seed_sql.strip().split(";"):
                s = stmt.strip()
                if s:
                    self._conn.execute(s)
            self._conn.commit()

        self._state = SqlDebugState(
            episode_id=episode_id or str(uuid.uuid4()),
            step_count=0,
            current_task_id=task_id,
            best_score=0.0,
        )

        return SqlDebugObservation(
            task_id=task_id,
            task_description=task.description,
            schema_info=task.ddl.strip(),
            broken_query=task.broken_query,
            explain_plan=task.explain_plan,
            reward=0.0,
            done=False,
            metadata={"message": "Episode started. Submit your SQL."},
        )

    def step(self, action: SqlDebugAction, timeout_s=None, **kwargs) -> SqlDebugObservation:
        """Execute agent's SQL, grade it, return observation with reward."""
        if self._conn is None or self._current_task is None:
            raise RuntimeError("Call reset() before step()")

        self._state.step_count += 1
        task = self._current_task
        grader = GRADER_MAP[task.task_id]

        score, grading_info = grader(action.sql_query, self._conn, task)

        # Step penalty: -0.05 per step beyond the first
        if self._state.step_count > 1:
            penalty = round(0.05 * (self._state.step_count - 1), 4)
            score = round(max(0.0, score - penalty), 4)
            grading_info["step_penalty"] = penalty

        if score > self._state.best_score:
            self._state.best_score = score

        done = score >= 0.95 or self._state.step_count >= MAX_STEPS

        # Get query result for observation
        query_result = None
        error_message = None
        try:
            cur = self._conn.cursor()
            cur.execute(action.sql_query)
            query_result = [list(r) for r in cur.fetchall()]
        except Exception as e:
            error_message = str(e)

        return SqlDebugObservation(
            task_id=task.task_id,
            task_description=task.description,
            schema_info=task.ddl.strip(),
            broken_query=task.broken_query,
            explain_plan=task.explain_plan,
            query_result=query_result,
            error_message=error_message,
            grading_info=grading_info,
            reward=score,
            done=done,
            metadata={"step": self._state.step_count, "best_score": self._state.best_score},
        )

    @property
    def state(self) -> SqlDebugState:
        return self._state

    def _seed_events(self, conn: sqlite3.Connection):
        """
        Generate 500 realistic event rows for task 3.

        Uses current date as base so that queries using date('now', '-30 days')
        return real results. The random seed (42) ensures the data distribution
        is deterministic, but the actual dates shift with wall-clock time
        (which is what we want — the task tests index usage, not memorised output).
        """
        rng = random.Random(42)
        event_types = ["purchase", "view", "click", "login", "logout"]
        base = datetime.now()  # Current date so 30-day window has real data
        rows = [
            (
                i,
                rng.randint(1, 30),
                rng.choices(event_types, weights=[20, 40, 25, 10, 5])[0],
                (base - timedelta(days=rng.randint(0, 45))).strftime("%Y-%m-%d"),
                None,
            )
            for i in range(1, 501)
        ]
        conn.executemany("INSERT INTO events VALUES (?,?,?,?,?)", rows)
        conn.commit()
