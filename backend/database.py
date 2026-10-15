"""
database.py - PostgreSQL connection and chat history helpers
"""

import os
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")


def get_connection():
    """Return a new psycopg2 connection using DATABASE_URL."""
    if not DATABASE_URL:
        raise RuntimeError(
            "DATABASE_URL is not set. Add it to your .env file.\n"
            "Example: DATABASE_URL=postgresql://user:password@host:5432/dbname"
        )
    return psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)


def init_db():
    """Create the chat_history table if it does not already exist."""
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS chat_history (
        id         SERIAL PRIMARY KEY,
        session_id TEXT        NOT NULL,
        role       TEXT        NOT NULL,
        content    TEXT        NOT NULL,
        created_at TIMESTAMPTZ DEFAULT NOW()
    );
    CREATE INDEX IF NOT EXISTS idx_chat_history_session
        ON chat_history (session_id, created_at);
    """
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(create_table_sql)
        conn.commit()
        print("✅ Database initialised — chat_history table ready.")
    finally:
        conn.close()


def save_message(session_id: str, role: str, content: str):
    """Persist a single chat turn (role = 'user' | 'assistant') to the DB."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO chat_history (session_id, role, content) VALUES (%s, %s, %s)",
                (session_id, role, content),
            )
        conn.commit()
    finally:
        conn.close()


def get_history(session_id: str, limit: int = 10) -> list[dict]:
    """
    Return the last `limit` turns for a session, oldest first.
    Each item is a dict with keys: role, content.
    """
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT role, content FROM (
                    SELECT role, content, created_at
                    FROM chat_history
                    WHERE session_id = %s
                    ORDER BY created_at DESC
                    LIMIT %s
                ) sub
                ORDER BY created_at ASC
                """,
                (session_id, limit),
            )
            rows = cur.fetchall()
        return [{"role": r["role"], "content": r["content"]} for r in rows]
    finally:
        conn.close()


def clear_session(session_id: str):
    """Delete all chat history rows for the given session."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM chat_history WHERE session_id = %s", (session_id,)
            )
        conn.commit()
    finally:
        conn.close()
