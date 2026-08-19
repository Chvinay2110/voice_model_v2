"""
db.py

Persistent SQLite database for storing meeting turns, registered speaker profiles,
avatars, and voiceprint embeddings.
"""

import base64
import json
import logging
import os
import sqlite3
import threading
from typing import Dict, List, Optional

log = logging.getLogger("db")

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "meetings.db")


_local = threading.local()


def get_conn():
    """Return a thread-local persistent connection (PRAGMAs run once per thread)."""
    conn = getattr(_local, 'conn', None)
    if conn is None:
        conn = sqlite3.connect(DB_PATH, timeout=20.0)
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA synchronous = NORMAL;")
        conn.row_factory = sqlite3.Row
        _local.conn = conn
    return conn


def init_db():
    with get_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS turns (
                turn_order INTEGER PRIMARY KEY,
                text TEXT,
                speaker_label TEXT,
                tagged_as TEXT,
                predicted_speaker TEXT,
                predicted_confidence REAL DEFAULT 0.0,
                is_mixed INTEGER DEFAULT 0,
                embedding_json TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS profiles (
                name TEXT PRIMARY KEY,
                avatar_b64 TEXT,
                company_name TEXT,
                samples_count INTEGER DEFAULT 0,
                centroid_json TEXT
            );
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS meeting_meta (
                key TEXT PRIMARY KEY,
                value_json TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        # Non-destructive migrations for existing databases
        try:
            conn.execute("ALTER TABLE turns ADD COLUMN embedding_json TEXT")
        except sqlite3.OperationalError:
            pass
        try:
            conn.execute("ALTER TABLE turns ADD COLUMN is_mixed INTEGER DEFAULT 0")
        except sqlite3.OperationalError:
            pass
        try:
            conn.execute("ALTER TABLE profiles ADD COLUMN company_name TEXT")
        except sqlite3.OperationalError:
            pass
        conn.commit()
    log.info("SQLite database initialized at %s (audio storage disabled)", DB_PATH)


# ── Turn Operations ──────────────────────────────────────────────────────────


def upsert_turn(
    turn_order: int,
    text: str,
    speaker_label: str = "A",
    audio_b64: Optional[str] = None,
    tagged_as: Optional[str] = None,
    predicted_speaker: Optional[str] = None,
    predicted_confidence: float = 0.0,
    embedding_json: Optional[str] = None,
    is_mixed: Optional[int] = None,
):
    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO turns (turn_order, text, speaker_label, tagged_as, predicted_speaker, predicted_confidence, embedding_json, is_mixed)
            VALUES (?, ?, ?, ?, ?, ?, ?, COALESCE(?, 0))
            ON CONFLICT(turn_order) DO UPDATE SET
                text = excluded.text,
                speaker_label = excluded.speaker_label,
                tagged_as = COALESCE(excluded.tagged_as, turns.tagged_as),
                predicted_speaker = COALESCE(excluded.predicted_speaker, turns.predicted_speaker),
                predicted_confidence = CASE WHEN excluded.predicted_confidence > 0 THEN excluded.predicted_confidence ELSE turns.predicted_confidence END,
                embedding_json = COALESCE(excluded.embedding_json, turns.embedding_json),
                is_mixed = CASE WHEN ? IS NOT NULL THEN ? ELSE turns.is_mixed END
        """,
            (
                turn_order,
                text,
                speaker_label,
                tagged_as,
                predicted_speaker,
                predicted_confidence,
                embedding_json,
                is_mixed,
                is_mixed,
                is_mixed,
            ),
        )
        conn.commit()


def update_turn_mixed(turn_order: int, is_mixed: int):
    """Update whether a turn is mixed audio / excluded from centroids."""
    with get_conn() as conn:
        conn.execute("UPDATE turns SET is_mixed = ? WHERE turn_order = ?", (is_mixed, turn_order))
        conn.commit()


def get_all_turns() -> List[dict]:
    """Full turn data including embeddings (used at startup)."""
    with get_conn() as conn:
        cur = conn.execute("SELECT * FROM turns ORDER BY turn_order DESC")
        rows = cur.fetchall()
        return [dict(r) for r in rows]


def get_all_turns_lite() -> List[dict]:
    """Lightweight turn listing without heavy embedding blobs."""
    with get_conn() as conn:
        cur = conn.execute(
            "SELECT turn_order, text, speaker_label, tagged_as, "
            "predicted_speaker, predicted_confidence, is_mixed, created_at, "
            "0 AS has_audio "
            "FROM turns ORDER BY turn_order DESC"
        )
        rows = cur.fetchall()
        return [dict(r) for r in rows]


def get_turn(turn_order: int) -> Optional[dict]:
    with get_conn() as conn:
        cur = conn.execute("SELECT * FROM turns WHERE turn_order = ?", (turn_order,))
        row = cur.fetchone()
        return dict(row) if row else None


def update_turn_tag(turn_order: int, tagged_as: Optional[str]):
    with get_conn() as conn:
        conn.execute("UPDATE turns SET tagged_as = ? WHERE turn_order = ?", (tagged_as, turn_order))
        conn.commit()


def update_turn_prediction_and_tag(
    turn_order: int,
    tagged_as: Optional[str] = None,
    predicted_speaker: Optional[str] = None,
    predicted_confidence: float = 0.0,
):
    with get_conn() as conn:
        conn.execute(
            """
            UPDATE turns
            SET tagged_as = ?,
                predicted_speaker = ?,
                predicted_confidence = ?
            WHERE turn_order = ?
            """,
            (tagged_as, predicted_speaker, predicted_confidence, turn_order),
        )
        conn.commit()


def batch_update_turn_predictions(updates: list):
    """Batch update predictions in a single transaction.

    Args:
        updates: list of (tagged_as, predicted_speaker, predicted_confidence, turn_order) tuples
    """
    if not updates:
        return
    with get_conn() as conn:
        conn.executemany(
            """
            UPDATE turns
            SET tagged_as = ?,
                predicted_speaker = ?,
                predicted_confidence = ?
            WHERE turn_order = ?
            """,
            updates,
        )
        conn.commit()


def update_turn_embedding(turn_order: int, embedding_json: str):
    """Store a computed embedding for a turn."""
    with get_conn() as conn:
        conn.execute(
            "UPDATE turns SET embedding_json = ? WHERE turn_order = ?",
            (embedding_json, turn_order),
        )
        conn.commit()


def delete_turn(turn_order: int):
    """Delete a single turn from the database."""
    with get_conn() as conn:
        conn.execute("DELETE FROM turns WHERE turn_order = ?", (turn_order,))
        conn.commit()


def clear_all_turn_tags():
    """Clear all speaker tags and predictions from turns in SQLite while keeping transcript text intact."""
    with get_conn() as conn:
        conn.execute("UPDATE turns SET tagged_as = NULL, predicted_speaker = NULL, predicted_confidence = 0.0")
        conn.commit()


def clear_all_turns():
    with get_conn() as conn:
        conn.execute("DELETE FROM turns")
        conn.commit()


# ── Profile Operations ───────────────────────────────────────────────────────


def upsert_profile(
    name: str,
    avatar_b64: Optional[str] = None,
    company_name: Optional[str] = None,
    samples_count: int = 0,
    centroid_list: Optional[list] = None,
):
    centroid_json = json.dumps(centroid_list) if centroid_list is not None else None
    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO profiles (name, avatar_b64, company_name, samples_count, centroid_json)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(name) DO UPDATE SET
                avatar_b64 = COALESCE(excluded.avatar_b64, profiles.avatar_b64),
                company_name = COALESCE(excluded.company_name, profiles.company_name),
                samples_count = excluded.samples_count,
                centroid_json = COALESCE(excluded.centroid_json, profiles.centroid_json)
        """,
            (name, avatar_b64, company_name, samples_count, centroid_json),
        )
        conn.commit()


def get_all_profiles() -> List[dict]:
    with get_conn() as conn:
        cur = conn.execute("SELECT name, avatar_b64, company_name, samples_count, centroid_json FROM profiles ORDER BY name ASC")
        rows = cur.fetchall()
        res = []
        for r in rows:
            d = dict(r)
            d["has_voiceprint"] = bool(d["centroid_json"])
            d["centroid"] = json.loads(d["centroid_json"]) if d["centroid_json"] else None
            del d["centroid_json"]
            res.append(d)
        return res


def delete_profile(name: str):
    with get_conn() as conn:
        conn.execute("DELETE FROM profiles WHERE LOWER(name) = LOWER(?)", (name,))
        conn.execute("UPDATE turns SET tagged_as = NULL WHERE LOWER(tagged_as) = LOWER(?)", (name,))
        conn.execute("UPDATE turns SET predicted_speaker = NULL, predicted_confidence = 0.0 WHERE LOWER(predicted_speaker) = LOWER(?)", (name,))
        conn.commit()


def clear_all_profiles():
    with get_conn() as conn:
        conn.execute("DELETE FROM profiles")
        conn.execute("DELETE FROM turns")
        conn.execute("DELETE FROM meeting_meta")
        conn.commit()


# ── Meeting Meta / Analytics Operations ──────────────────────────────────────


def set_meeting_meta(key: str, value):
    """Save persistent JSON data for wordcloud, speaker analysis, or meeting summary."""
    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO meeting_meta (key, value_json, updated_at)
            VALUES (?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(key) DO UPDATE SET
                value_json = excluded.value_json,
                updated_at = CURRENT_TIMESTAMP
            """,
            (key, json.dumps(value)),
        )
        conn.commit()


def get_meeting_meta(key: str, default=None):
    """Retrieve persistent JSON data for a given key."""
    with get_conn() as conn:
        cur = conn.execute("SELECT value_json FROM meeting_meta WHERE key = ?", (key,))
        row = cur.fetchone()
        if row and row["value_json"]:
            try:
                return json.loads(row["value_json"])
            except Exception:
                return default
        return default


def clear_meeting_meta():
    """Clear all persistent analytics metadata."""
    with get_conn() as conn:
        conn.execute("DELETE FROM meeting_meta")
        conn.commit()
