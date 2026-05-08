"""
SQLite-backed analysis cache for the YouTube Content Analyzer.
Stores complete analysis results keyed by video ID so repeat requests
return instantly without re-running the ML pipeline.
"""

import json
import os
import sqlite3
import logging
from datetime import datetime, timezone
from typing import Optional, List, Dict

logger = logging.getLogger(__name__)

DB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
DB_PATH = os.path.join(DB_DIR, 'analysis_history.db')

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS analysis_history (
    video_id      TEXT PRIMARY KEY,
    url           TEXT NOT NULL,
    title         TEXT NOT NULL DEFAULT '',
    channel       TEXT NOT NULL DEFAULT '',
    thumbnail_url TEXT NOT NULL DEFAULT '',
    result_json   TEXT NOT NULL,
    analyzed_at   TEXT NOT NULL
);
"""


class AnalysisCache:
    """Thread-safe, file-based analysis cache using SQLite."""

    def __init__(self, db_path: str = DB_PATH):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._db_path = db_path
        self._init_db()
        logger.info(f"AnalysisCache initialised — DB at {db_path}")

    def _conn(self) -> sqlite3.Connection:
        """Return a new connection (safe for multi-threaded Flask)."""
        conn = sqlite3.connect(self._db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._conn() as conn:
            conn.execute(_CREATE_TABLE)
            conn.commit()

    def get(self, video_id: str) -> Optional[dict]:
        """
        Retrieve cached analysis result for a video.

        Returns:
            The full result dict (as originally returned by
            ``GlobalOrchestrator.get_results_as_dict``) with an extra
            ``"cached": True`` flag, or ``None`` if not found.
        """
        with self._conn() as conn:
            row = conn.execute(
                "SELECT result_json, analyzed_at FROM analysis_history WHERE video_id = ?",
                (video_id,),
            ).fetchone()
        if row is None:
            return None

        result = json.loads(row["result_json"])
        result["cached"] = True
        result["analyzed_at"] = row["analyzed_at"]
        return result

    def put(self, video_id: str, url: str, result: dict) -> None:
        """Store (or overwrite) an analysis result."""
        video_section = result.get("video", {})
        title = video_section.get("title", "")
        channel = video_section.get("channel", "")
        thumbnail = video_section.get("thumbnail_url", "")
        now = datetime.now(timezone.utc).isoformat()

        with self._conn() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO analysis_history
                    (video_id, url, title, channel, thumbnail_url, result_json, analyzed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (video_id, url, title, channel, thumbnail, json.dumps(result), now),
            )
            conn.commit()
        logger.info(f"Cached analysis for {video_id} ({title})")

    def list_history(self, limit: int = 50) -> List[Dict]:
        """Return a summary list of cached analyses (most recent first)."""
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT video_id, url, title, channel, thumbnail_url, analyzed_at
                FROM analysis_history
                ORDER BY analyzed_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def delete(self, video_id: str) -> bool:
        """Delete a cached entry. Returns True if something was deleted."""
        with self._conn() as conn:
            cursor = conn.execute(
                "DELETE FROM analysis_history WHERE video_id = ?",
                (video_id,),
            )
            conn.commit()
        deleted = cursor.rowcount > 0
        if deleted:
            logger.info(f"Deleted cache entry for {video_id}")
        return deleted
