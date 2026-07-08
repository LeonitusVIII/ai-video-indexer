"""Batched SQLite catalog writes for pipeline scripts."""
import datetime
import json
import sqlite3
from pathlib import Path

from job_utils import (
    compute_file_fingerprint,
    ensure_video_thumbnail,
    get_video_duration,
    metadata_json_path,
    transcript_json_path,
    vision_json_path,
)
from pipeline_utils import vision_output_complete

CATALOG_COMMIT_BATCH = 50


class CatalogWriter:
    """Reuse one SQLite connection and commit in batches during pipeline loops."""

    def __init__(self, db_file, batch_size=CATALOG_COMMIT_BATCH):
        self.db_file = str(db_file)
        self.batch_size = max(1, int(batch_size))
        self.con = sqlite3.connect(self.db_file)
        self.pending = 0

    def execute(self, sql, params=()):
        self.con.execute(sql, params)
        self.pending += 1
        self._commit_if_needed()

    def _commit_if_needed(self, force=False):
        if force or self.pending >= self.batch_size:
            self.con.commit()
            self.pending = 0

    def close(self):
        self._commit_if_needed(force=True)
        self.con.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is None:
            self.close()
        else:
            try:
                self.con.rollback()
            except Exception:
                pass
            self.con.close()
        return False


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


def sidecar_flags_for_video(video_path):
    path = Path(video_path)
    return {
        "has_transcript": int(transcript_json_path(path).exists()),
        "has_vision": int(vision_output_complete(vision_json_path(path))),
        "has_metadata": int(metadata_json_path(path).exists()),
    }


def update_video_flag(writer, video_path, column):
    writer.execute(
        f"UPDATE videos SET {column} = 1 WHERE path = ?",
        (str(video_path),),
    )


def update_video_transcript(writer, video_path, language=None):
    if language:
        writer.execute(
            "UPDATE videos SET has_transcript = 1, transcript_language = ? WHERE path = ?",
            (str(language), str(video_path)),
        )
    else:
        writer.execute(
            "UPDATE videos SET has_transcript = 1 WHERE path = ?",
            (str(video_path),),
        )


def update_video_metadata(writer, video_path, people_tags=None):
    if people_tags:
        writer.execute(
            """
            UPDATE videos SET
                has_metadata = 1,
                people_tags = ?
            WHERE path = ?
            """,
            (json.dumps(people_tags or []), str(video_path)),
        )
    else:
        update_video_flag(writer, video_path, "has_metadata")


def upsert_scanned_video(writer, video_path, *, scan_meta=True):
    """Insert or update a catalog row and sync sidecar flags in one transaction batch."""
    video_path = Path(video_path)
    stat = video_path.stat()
    duration_seconds = None
    file_fingerprint = None
    if scan_meta:
        try:
            duration_seconds = round(get_video_duration(video_path), 3)
        except Exception:
            duration_seconds = None
        try:
            file_fingerprint = compute_file_fingerprint(video_path, stat.st_size)
        except Exception:
            file_fingerprint = None
        try:
            ensure_video_thumbnail(video_path, duration_seconds)
        except Exception:
            pass

    writer.execute(
        """
        INSERT INTO videos (
            path,
            filename,
            folder,
            extension,
            size_bytes,
            modified_time,
            scanned_at,
            duration_seconds,
            file_fingerprint
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(path) DO UPDATE SET
            filename=excluded.filename,
            folder=excluded.folder,
            extension=excluded.extension,
            size_bytes=excluded.size_bytes,
            modified_time=excluded.modified_time,
            scanned_at=excluded.scanned_at,
            duration_seconds=COALESCE(excluded.duration_seconds, videos.duration_seconds),
            file_fingerprint=COALESCE(excluded.file_fingerprint, videos.file_fingerprint)
        """,
        (
            str(video_path),
            video_path.name,
            str(video_path.parent),
            video_path.suffix.lower(),
            stat.st_size,
            datetime.datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
            datetime.datetime.now().isoformat(timespec="seconds"),
            duration_seconds,
            file_fingerprint,
        ),
    )

    flags = sidecar_flags_for_video(video_path)
    language = _language_from_transcript_sidecar(video_path)
    people_json = _people_tags_from_metadata_sidecar(video_path)
    writer.execute(
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


def delete_video_paths(writer, paths):
    for path in paths:
        writer.execute("DELETE FROM videos WHERE path = ?", (str(path),))
