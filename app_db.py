"""SQLite catalog helpers for AI Video Indexer."""
import sqlite3
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent / "scripts"
import sys

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from job_utils import metadata_json_path, transcript_json_path, vision_json_path

DB_FILE = Path(__file__).resolve().parent / "data" / "video_indexer.db"


def migrate_db_schema(con):
    """Apply lightweight schema migrations for existing databases."""
    cur = con.cursor()
    cur.execute("PRAGMA table_info(videos)")
    columns = {row[1] for row in cur.fetchall()}
    if "has_enhanced_audio" in columns:
        try:
            cur.execute("ALTER TABLE videos DROP COLUMN has_enhanced_audio")
        except sqlite3.OperationalError:
            pass
    con.commit()


def init_db(db_file=None):
    db_path = Path(db_file or DB_FILE)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS videos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            path TEXT UNIQUE,
            filename TEXT,
            folder TEXT,
            extension TEXT,
            size_bytes INTEGER,
            modified_time TEXT,
            scanned_at TEXT,
            has_transcript INTEGER DEFAULT 0,
            has_vision INTEGER DEFAULT 0,
            has_metadata INTEGER DEFAULT 0,
            indexed_in_qdrant INTEGER DEFAULT 0
        )
    """)
    con.commit()
    migrate_db_schema(con)
    cur.execute("PRAGMA journal_mode=WAL")
    con.commit()
    con.close()


def get_videos(folder=None, db_file=None):
    db_path = Path(db_file or DB_FILE)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    cur = con.cursor()
    if folder:
        cur.execute(
            "SELECT * FROM videos WHERE path LIKE ? ORDER BY path",
            (folder.rstrip("\\/") + "%",),
        )
    else:
        cur.execute("SELECT * FROM videos ORDER BY path")
    rows = [dict(r) for r in cur.fetchall()]
    con.close()
    return rows


def get_library_stats(db_file=None):
    con = sqlite3.connect(db_file or DB_FILE)
    cur = con.cursor()
    cur.execute("""
        SELECT
            COUNT(*) AS total,
            COALESCE(SUM(has_transcript), 0),
            COALESCE(SUM(has_vision), 0),
            COALESCE(SUM(indexed_in_qdrant), 0)
        FROM videos
    """)
    row = cur.fetchone()
    con.close()
    return {
        "total": int(row[0]),
        "transcribed": int(row[1]),
        "vision": int(row[2]),
        "indexed": int(row[3]),
    }


def sidecar_flags_for_video(video_path):
    path = Path(video_path)
    return {
        "has_transcript": int(transcript_json_path(path).exists()),
        "has_vision": int(vision_json_path(path).exists()),
        "has_metadata": int(metadata_json_path(path).exists()),
    }


def sync_sidecar_flags(db_file, video_path):
    flags = sidecar_flags_for_video(video_path)
    con = sqlite3.connect(db_file)
    cur = con.cursor()
    cur.execute(
        """
        UPDATE videos SET
            has_transcript = ?,
            has_vision = ?,
            has_metadata = ?
        WHERE path = ?
        """,
        (
            flags["has_transcript"],
            flags["has_vision"],
            flags["has_metadata"],
            str(video_path),
        ),
    )
    con.commit()
    con.close()
    return flags


def prune_missing_videos(db_file, folder_prefix=None):
    """Remove catalog rows whose files no longer exist on disk."""
    con = sqlite3.connect(db_file)
    cur = con.cursor()
    if folder_prefix:
        cur.execute(
            "SELECT path FROM videos WHERE path LIKE ?",
            (str(folder_prefix).rstrip("\\/") + "%",),
        )
    else:
        cur.execute("SELECT path FROM videos")
    removed = []
    for (path,) in cur.fetchall():
        if not Path(path).exists():
            cur.execute("DELETE FROM videos WHERE path = ?", (path,))
            removed.append(path)
    con.commit()
    con.close()
    return removed


def remove_folder_from_library(db_file, folder, *, qdrant_cleanup_fn=None):
    prefix = folder.rstrip("\\/") + "%"
    con = sqlite3.connect(db_file)
    cur = con.cursor()
    cur.execute("DELETE FROM videos WHERE path LIKE ?", (prefix,))
    deleted = cur.rowcount
    con.commit()
    con.close()
    if qdrant_cleanup_fn:
        qdrant_cleanup_fn(folder)
    return deleted
