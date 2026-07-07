"""SQLite catalog helpers for AI Video Indexer."""
import json
import sqlite3
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent / "scripts"
import sys

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from job_utils import metadata_json_path, transcript_json_path, vision_json_path
from pipeline_utils import vision_output_complete

DB_FILE = Path(__file__).resolve().parent / "data" / "video_indexer.db"

SCAN_META_COLUMNS = {
    "duration_seconds": "REAL",
    "file_fingerprint": "TEXT",
    "transcript_language": "TEXT",
    "people_tags": "TEXT",
}


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
    for name, col_type in SCAN_META_COLUMNS.items():
        if name not in columns:
            cur.execute(f"ALTER TABLE videos ADD COLUMN {name} {col_type}")
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
            indexed_in_qdrant INTEGER DEFAULT 0,
            duration_seconds REAL,
            file_fingerprint TEXT,
            transcript_language TEXT,
            people_tags TEXT
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
        "has_vision": int(vision_output_complete(vision_json_path(path))),
        "has_metadata": int(metadata_json_path(path).exists()),
    }


def _language_from_transcript_sidecar(video_path):
    path = transcript_json_path(Path(video_path))
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    lang = data.get("language")
    return str(lang) if lang else None


def _people_tags_from_metadata_sidecar(video_path):
    path = metadata_json_path(Path(video_path))
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    tags = data.get("people_tags") or []
    if not tags:
        return None
    return json.dumps(tags)


def sync_sidecar_flags(db_file, video_path):
    flags = sidecar_flags_for_video(video_path)
    language = _language_from_transcript_sidecar(video_path)
    people_json = _people_tags_from_metadata_sidecar(video_path)
    con = sqlite3.connect(db_file)
    cur = con.cursor()
    cur.execute(
        """
        UPDATE videos SET
            has_transcript = ?,
            has_vision = ?,
            has_metadata = ?,
            transcript_language = COALESCE(?, transcript_language),
            people_tags = COALESCE(?, people_tags)
        WHERE path = ?
        """,
        (
            flags["has_transcript"],
            flags["has_vision"],
            flags["has_metadata"],
            language,
            people_json,
            str(video_path),
        ),
    )
    con.commit()
    con.close()
    return flags


def update_video_scan_meta(db_file, video_path, *, duration_seconds=None, file_fingerprint=None):
    con = sqlite3.connect(db_file)
    cur = con.cursor()
    cur.execute(
        """
        UPDATE videos SET
            duration_seconds = COALESCE(?, duration_seconds),
            file_fingerprint = COALESCE(?, file_fingerprint)
        WHERE path = ?
        """,
        (duration_seconds, file_fingerprint, str(video_path)),
    )
    con.commit()
    con.close()


def update_video_transcript_language(db_file, video_path, language):
    if not language:
        return
    con = sqlite3.connect(db_file)
    cur = con.cursor()
    cur.execute(
        "UPDATE videos SET has_transcript = 1, transcript_language = ? WHERE path = ?",
        (str(language), str(video_path)),
    )
    con.commit()
    con.close()


def update_video_people_tags(db_file, video_path, tags):
    con = sqlite3.connect(db_file)
    cur = con.cursor()
    cur.execute(
        "UPDATE videos SET people_tags = ? WHERE path = ?",
        (json.dumps(tags or []), str(video_path)),
    )
    con.commit()
    con.close()


def find_duplicate_groups(db_file=None, *, folder=None):
    """Group videos that likely duplicate each other (fingerprint or size+duration)."""
    videos = get_videos(folder=folder, db_file=db_file)
    by_fingerprint = {}
    by_size_duration = {}

    for video in videos:
        path = video.get("path")
        if not path:
            continue
        fp = video.get("file_fingerprint")
        if fp:
            by_fingerprint.setdefault(fp, []).append(video)
            continue
        size = video.get("size_bytes")
        duration = video.get("duration_seconds")
        if size and duration:
            key = (int(size), round(float(duration), 1))
            by_size_duration.setdefault(key, []).append(video)

    groups = []
    seen_paths = set()
    for bucket in (by_fingerprint, by_size_duration):
        for group in bucket.values():
            if len(group) < 2:
                continue
            paths = tuple(sorted(v["path"] for v in group))
            if paths in seen_paths:
                continue
            seen_paths.add(paths)
            match_type = "fingerprint" if bucket is by_fingerprint else "size+duration"
            groups.append({"match_type": match_type, "videos": group})
    groups.sort(key=lambda g: g["videos"][0].get("filename", ""))
    return groups


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
