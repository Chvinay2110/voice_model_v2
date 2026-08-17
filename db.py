"""
db.py

Persistent SQLite database for storing meeting turns, registered speaker profiles,
avatars, and voiceprint embeddings.
"""

import json
import logging
import os
import sqlite3
from typing import Dict, List, Optional

log = logging.getLogger("db")

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "meetings.db")


def get_conn():
    conn = sqlite3.connect(DB_PATH, timeout=10.0)
    conn.row_factory = sqlite3.Row
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
                audio_b64 TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS profiles (
                name TEXT PRIMARY KEY,
                avatar_b64 TEXT,
                samples_count INTEGER DEFAULT 0,
                centroid_json TEXT
            );
        """)
        conn.commit()
    log.info("SQLite database initialized at %s", DB_PATH)


# ── Turn Operations ──────────────────────────────────────────────────────────


def upsert_turn(
    turn_order: int,
    text: str,
    speaker_label: str = "A",
    audio_b64: Optional[str] = None,
    tagged_as: Optional[str] = None,
    predicted_speaker: Optional[str] = None,
    predicted_confidence: float = 0.0,
):
    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO turns (turn_order, text, speaker_label, audio_b64, tagged_as, predicted_speaker, predicted_confidence)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(turn_order) DO UPDATE SET
                text = excluded.text,
                speaker_label = excluded.speaker_label,
                audio_b64 = COALESCE(excluded.audio_b64, turns.audio_b64),
                tagged_as = COALESCE(excluded.tagged_as, turns.tagged_as),
                predicted_speaker = COALESCE(excluded.predicted_speaker, turns.predicted_speaker),
                predicted_confidence = CASE WHEN excluded.predicted_confidence > 0 THEN excluded.predicted_confidence ELSE turns.predicted_confidence END
        """,
            (
                turn_order,
                text,
                speaker_label,
                audio_b64,
                tagged_as,
                predicted_speaker,
                predicted_confidence,
            ),
        )
        conn.commit()


def get_all_turns() -> List[dict]:
    with get_conn() as conn:
        cur = conn.execute("SELECT * FROM turns ORDER BY turn_order DESC")
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


def clear_all_turns():
    with get_conn() as conn:
        conn.execute("DELETE FROM turns")
        conn.commit()


# ── Profile Operations ───────────────────────────────────────────────────────


def upsert_profile(
    name: str,
    avatar_b64: Optional[str] = None,
    samples_count: int = 0,
    centroid_list: Optional[list] = None,
):
    centroid_json = json.dumps(centroid_list) if centroid_list is not None else None
    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO profiles (name, avatar_b64, samples_count, centroid_json)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(name) DO UPDATE SET
                avatar_b64 = COALESCE(excluded.avatar_b64, profiles.avatar_b64),
                samples_count = excluded.samples_count,
                centroid_json = COALESCE(excluded.centroid_json, profiles.centroid_json)
        """,
            (name, avatar_b64, samples_count, centroid_json),
        )
        conn.commit()


def get_all_profiles() -> List[dict]:
    with get_conn() as conn:
        cur = conn.execute("SELECT name, avatar_b64, samples_count, centroid_json FROM profiles ORDER BY name ASC")
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
        conn.execute("DELETE FROM profiles WHERE name = ?", (name,))
        conn.execute("UPDATE turns SET tagged_as = NULL WHERE LOWER(tagged_as) = LOWER(?)", (name,))
        conn.commit()


def clear_all_profiles():
    with get_conn() as conn:
        conn.execute("DELETE FROM profiles")
        conn.execute("DELETE FROM turns")
        conn.commit()
