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
                audio_b64 TEXT,
                embedding_json TEXT,
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
        # Migration: add embedding_json column to existing databases
        try:
            conn.execute("ALTER TABLE turns ADD COLUMN embedding_json TEXT")
        except sqlite3.OperationalError:
            pass  # Column already exists
        # Migration: add audio_blob BLOB column and convert existing base64 TEXT data
        try:
            conn.execute("ALTER TABLE turns ADD COLUMN audio_blob BLOB")
            # One-time migration: convert base64 TEXT → raw BLOB
            rows = conn.execute(
                "SELECT turn_order, audio_b64 FROM turns WHERE audio_b64 IS NOT NULL AND audio_blob IS NULL"
            ).fetchall()
            for row in rows:
                try:
                    audio_bytes = base64.b64decode(row["audio_b64"])
                    conn.execute(
                        "UPDATE turns SET audio_blob = ?, audio_b64 = NULL WHERE turn_order = ?",
                        (audio_bytes, row["turn_order"]),
                    )
                except Exception:
                    pass
            log.info("Migrated %d audio turns from base64 TEXT to BLOB.", len(rows))
        except sqlite3.OperationalError:
            # Column already exists — migrate any remaining TEXT entries
            rows = conn.execute(
                "SELECT turn_order, audio_b64 FROM turns WHERE audio_b64 IS NOT NULL AND audio_blob IS NULL"
            ).fetchall()
            for row in rows:
                try:
                    audio_bytes = base64.b64decode(row["audio_b64"])
                    conn.execute(
                        "UPDATE turns SET audio_blob = ?, audio_b64 = NULL WHERE turn_order = ?",
                        (audio_bytes, row["turn_order"]),
                    )
                except Exception:
                    pass
            if rows:
                log.info("Migrated %d remaining audio turns from TEXT to BLOB.", len(rows))
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
    embedding_json: Optional[str] = None,
):
    # Convert base64 text to raw bytes for compact BLOB storage
    audio_blob = base64.b64decode(audio_b64) if audio_b64 else None
    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO turns (turn_order, text, speaker_label, audio_blob, tagged_as, predicted_speaker, predicted_confidence, embedding_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(turn_order) DO UPDATE SET
                text = excluded.text,
                speaker_label = excluded.speaker_label,
                audio_blob = COALESCE(excluded.audio_blob, turns.audio_blob),
                tagged_as = COALESCE(excluded.tagged_as, turns.tagged_as),
                predicted_speaker = COALESCE(excluded.predicted_speaker, turns.predicted_speaker),
                predicted_confidence = CASE WHEN excluded.predicted_confidence > 0 THEN excluded.predicted_confidence ELSE turns.predicted_confidence END,
                embedding_json = COALESCE(excluded.embedding_json, turns.embedding_json)
        """,
            (
                turn_order,
                text,
                speaker_label,
                audio_blob,
                tagged_as,
                predicted_speaker,
                predicted_confidence,
                embedding_json,
            ),
        )
        conn.commit()


def get_all_turns() -> List[dict]:
    """Full turn data including audio and embeddings (used at startup)."""
    with get_conn() as conn:
        cur = conn.execute("SELECT * FROM turns ORDER BY turn_order DESC")
        rows = cur.fetchall()
        result = []
        for r in rows:
            d = dict(r)
            # Convert BLOB back to base64 for in-memory compatibility
            if d.get("audio_blob"):
                d["audio_b64"] = base64.b64encode(d["audio_blob"]).decode("ascii")
            d.pop("audio_blob", None)
            result.append(d)
        return result


def get_all_turns_lite() -> List[dict]:
    """Lightweight turn listing without heavy audio/embedding blobs."""
    with get_conn() as conn:
        cur = conn.execute(
            "SELECT turn_order, text, speaker_label, tagged_as, "
            "predicted_speaker, predicted_confidence, created_at, "
            "(audio_blob IS NOT NULL OR audio_b64 IS NOT NULL) AS has_audio "
            "FROM turns ORDER BY turn_order DESC"
        )
        rows = cur.fetchall()
        return [dict(r) for r in rows]


def get_turn(turn_order: int) -> Optional[dict]:
    with get_conn() as conn:
        cur = conn.execute("SELECT * FROM turns WHERE turn_order = ?", (turn_order,))
        row = cur.fetchone()
        if not row:
            return None
        d = dict(row)
        if d.get("audio_blob"):
            d["audio_b64"] = base64.b64encode(d["audio_blob"]).decode("ascii")
        d.pop("audio_blob", None)
        return d


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
        conn.execute("DELETE FROM profiles WHERE LOWER(name) = LOWER(?)", (name,))
        conn.execute("UPDATE turns SET tagged_as = NULL WHERE LOWER(tagged_as) = LOWER(?)", (name,))
        conn.execute("UPDATE turns SET predicted_speaker = NULL, predicted_confidence = 0.0 WHERE LOWER(predicted_speaker) = LOWER(?)", (name,))
        conn.commit()


def clear_all_profiles():
    with get_conn() as conn:
        conn.execute("DELETE FROM profiles")
        conn.execute("DELETE FROM turns")
        conn.commit()
