import datetime
import gc
import json
import re
import sqlite3
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from job_utils import COLLECTION_NAME, EMBEDDING_MODEL, QDRANT_DIR, format_timestamp
from ml_bootstrap import prepare_ml_environment

APP_DIR = Path(__file__).resolve().parent
DB_FILE = APP_DIR / "data" / "video_indexer.db"
JOBS_DIR = APP_DIR / "jobs"
LOG_DIR = APP_DIR / "logs"
SEARCH_LOG_FILE = LOG_DIR / "search.log"

_PIPELINE_STEP_SUFFIXES = (
    "_scan",
    "_normalize",
    "_transcribe",
    "_vision",
    "_metadata",
    "_index",
)

_embedder = None


class QdrantLockError(Exception):
    """Local Qdrant storage is locked by another client (usually an indexing job)."""


def _qdrant_locked_message(exc):
    return isinstance(exc, RuntimeError) and "already accessed" in str(exc).lower()


def _qdrant_likely_busy():
    """True when a running job is likely holding the local Qdrant file lock."""
    if not JOBS_DIR.exists():
        return False

    steps_dir = JOBS_DIR / "steps"
    for path in JOBS_DIR.glob("*.json"):
        if any(path.stem.endswith(suffix) for suffix in _PIPELINE_STEP_SUFFIXES):
            continue
        try:
            job = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if job.get("status") != "running":
            continue
        script = job.get("script", "")
        if script == "index_qdrant.py":
            return True
        if script != "run_pipeline.py":
            continue
        index_step = steps_dir / f"{path.stem}_index.json"
        if not index_step.exists():
            continue
        try:
            step = json.loads(index_step.read_text(encoding="utf-8"))
        except Exception:
            continue
        if step.get("status") == "running":
            return True
    return False


def _open_qdrant_client():
    if not QDRANT_DIR.exists():
        return None
    from qdrant_client import QdrantClient
    try:
        return QdrantClient(path=str(QDRANT_DIR))
    except RuntimeError as exc:
        if _qdrant_locked_message(exc):
            raise QdrantLockError(str(exc)) from exc
        raise


def _open_qdrant_client_with_retry(max_attempts=12, delay_seconds=0.3):
    """Open local Qdrant with short retries so search can run between indexing videos."""
    last_exc = None
    for attempt in range(max_attempts):
        try:
            return _open_qdrant_client()
        except QdrantLockError as exc:
            last_exc = exc
            if attempt + 1 >= max_attempts:
                break
            time.sleep(delay_seconds)
    if last_exc is not None:
        raise last_exc
    return None


def _try_open_qdrant_for_read():
    """Best-effort read client; returns (client, locked) where locked means indexing holds the DB."""
    try:
        return _open_qdrant_client_with_retry(), False
    except QdrantLockError:
        return None, True


def _close_qdrant_client(client):
    if client is None:
        return
    try:
        client.close()
    except Exception:
        pass
    gc.collect()


def release_qdrant_client():
    """Hint GC to release any lingering local Qdrant file locks before subprocess jobs."""
    gc.collect()


def _get_embedder():
    global _embedder
    if _embedder is None:
        prepare_ml_environment()
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer(EMBEDDING_MODEL)
    return _embedder


def _catalog_paths(db_file=None):
    db_path = Path(db_file or DB_FILE)
    if not db_path.exists():
        return set()
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    cur.execute("SELECT path FROM videos")
    paths = {row[0] for row in cur.fetchall()}
    con.close()
    return paths


def get_search_index_stats(db_file=None):
    stats = {
        "collection_exists": False,
        "segment_count": 0,
        "catalog_videos": 0,
        "indexed_flags": 0,
        "qdrant_dir": str(QDRANT_DIR),
        "qdrant_locked": False,
    }
    db_path = Path(db_file or DB_FILE)
    if db_path.exists():
        con = sqlite3.connect(db_path)
        cur = con.cursor()
        cur.execute("SELECT COUNT(*) FROM videos")
        stats["catalog_videos"] = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM videos WHERE indexed_in_qdrant = 1")
        stats["indexed_flags"] = cur.fetchone()[0]
        con.close()

    if _qdrant_likely_busy():
        stats["qdrant_indexing_active"] = True

    client, locked = _try_open_qdrant_for_read()
    if locked:
        stats["qdrant_locked"] = True
        return stats

    if client is None:
        return stats
    try:
        if client.collection_exists(COLLECTION_NAME):
            stats["collection_exists"] = True
            info = client.get_collection(COLLECTION_NAME)
            stats["segment_count"] = int(info.points_count or 0)
    finally:
        _close_qdrant_client(client)
    return stats


def reset_search_index(db_file=None):
    """Delete all Qdrant segments and clear indexed flags in SQLite."""
    if _qdrant_likely_busy():
        return False, (
            "Cannot reset the search index while an indexing job is running. "
            "Wait for the job to finish or stop it first."
        )

    db_path = Path(db_file or DB_FILE)
    try:
        client = _open_qdrant_client()
    except QdrantLockError:
        return False, (
            "Cannot reset the search index while an indexing job is running. "
            "Wait for the job to finish or stop it first."
        )

    try:
        if client and client.collection_exists(COLLECTION_NAME):
            client.delete_collection(COLLECTION_NAME)
    finally:
        _close_qdrant_client(client)

    if db_path.exists():
        con = sqlite3.connect(db_path)
        cur = con.cursor()
        cur.execute("UPDATE videos SET indexed_in_qdrant = 0")
        con.commit()
        con.close()

    return True, (
        "Search index cleared. Run **Index search DB** on the Run Jobs tab to rebuild."
    )


def _folder_path_prefix(folder):
    return str(folder).rstrip("\\/")


def delete_qdrant_points_for_folder(folder):
    """Delete all Qdrant segments for videos under a folder path."""
    if not QDRANT_DIR.exists():
        return 0
    prefix = _folder_path_prefix(folder).lower()
    try:
        client = _open_qdrant_client()
    except QdrantLockError:
        return 0
    if client is None or not client.collection_exists(COLLECTION_NAME):
        _close_qdrant_client(client)
        return 0
    removed = 0
    try:
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        offset = None
        while True:
            records, offset = client.scroll(
                collection_name=COLLECTION_NAME,
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            if not records:
                break
            for record in records:
                path_str = (record.payload or {}).get("video_path", "")
                if not path_str:
                    continue
                normalized = path_str.rstrip("\\/").lower()
                if not (normalized == prefix or normalized.startswith(prefix + "\\") or normalized.startswith(prefix + "/")):
                    continue
                client.delete(
                    collection_name=COLLECTION_NAME,
                    points_selector=Filter(
                        must=[
                            FieldCondition(
                                key="video_path",
                                match=MatchValue(value=path_str),
                            )
                        ]
                    ),
                )
                removed += 1
            if offset is None:
                break
    finally:
        _close_qdrant_client(client)
    return removed


def rename_qdrant_video_path(old_path, new_path):
    """Update Qdrant payloads when a video path changes (e.g. after normalize)."""
    if not QDRANT_DIR.exists() or str(old_path) == str(new_path):
        return False
    try:
        client = _open_qdrant_client()
    except QdrantLockError:
        return False
    if client is None or not client.collection_exists(COLLECTION_NAME):
        _close_qdrant_client(client)
        return False
    try:
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        new_path_str = str(new_path)
        client.set_payload(
            collection_name=COLLECTION_NAME,
            payload={
                "video_path": new_path_str,
                "filename": Path(new_path_str).name,
                "folder": str(Path(new_path_str).parent),
            },
            points=Filter(
                must=[
                    FieldCondition(
                        key="video_path",
                        match=MatchValue(value=str(old_path)),
                    )
                ]
            ),
        )
        return True
    finally:
        _close_qdrant_client(client)


def prune_orphan_qdrant_points(db_file=None):
    """Remove Qdrant points whose video_path is not in the catalog or missing on disk."""
    if not QDRANT_DIR.exists():
        return 0
    catalog = _catalog_paths(db_file)
    try:
        client = _open_qdrant_client()
    except QdrantLockError:
        return 0
    if client is None or not client.collection_exists(COLLECTION_NAME):
        _close_qdrant_client(client)
        return 0
    removed = 0
    try:
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        offset = None
        while True:
            records, offset = client.scroll(
                collection_name=COLLECTION_NAME,
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            if not records:
                break
            for record in records:
                path_str = (record.payload or {}).get("video_path", "")
                if not path_str:
                    continue
                if catalog and path_str not in catalog:
                    client.delete(
                        collection_name=COLLECTION_NAME,
                        points_selector=Filter(
                            must=[
                                FieldCondition(
                                    key="video_path",
                                    match=MatchValue(value=path_str),
                                )
                            ]
                        ),
                    )
                    removed += 1
                elif path_str and not Path(path_str).exists():
                    client.delete(
                        collection_name=COLLECTION_NAME,
                        points_selector=Filter(
                            must=[
                                FieldCondition(
                                    key="video_path",
                                    match=MatchValue(value=path_str),
                                )
                            ]
                        ),
                    )
                    removed += 1
            if offset is None:
                break
    finally:
        _close_qdrant_client(client)
    return removed


def _date_to_timestamp(date_value, end_of_day=False):
    if not date_value:
        return None
    if isinstance(date_value, datetime.date) and not isinstance(date_value, datetime.datetime):
        dt = datetime.datetime.combine(date_value, datetime.time.max if end_of_day else datetime.time.min)
    else:
        dt = date_value
    return dt.timestamp()


def _tokenize(text):
    return [t for t in re.findall(r"[a-z0-9']+", (text or "").lower()) if len(t) > 1]


def _keyword_overlap(query, haystack):
    terms = _tokenize(query)
    if not terms:
        return 0.0
    text = haystack.lower()
    hits = sum(1 for term in terms if term in text)
    return hits / len(terms)


def _build_qdrant_filter(date_from=None, date_to=None, min_size_bytes=None, max_size_bytes=None):
    from qdrant_client.models import Filter, FieldCondition, Range

    conditions = []

    from_ts = _date_to_timestamp(date_from, end_of_day=False)
    to_ts = _date_to_timestamp(date_to, end_of_day=True)
    if from_ts is not None or to_ts is not None:
        range_kwargs = {}
        if from_ts is not None:
            range_kwargs["gte"] = from_ts
        if to_ts is not None:
            range_kwargs["lte"] = to_ts
        conditions.append(
            FieldCondition(key="modified_ts", range=Range(**range_kwargs))
        )

    if min_size_bytes is not None or max_size_bytes is not None:
        size_range = {}
        if min_size_bytes is not None:
            size_range["gte"] = int(min_size_bytes)
        if max_size_bytes is not None:
            size_range["lte"] = int(max_size_bytes)
        conditions.append(
            FieldCondition(key="size_bytes", range=Range(**size_range))
        )

    if not conditions:
        return None

    return Filter(must=conditions)


def _new_search_log(query, **filters):
    return {
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "query": query,
        "filters": filters,
        "steps": [],
        "index_stats": {},
        "qdrant_raw_hits": 0,
        "filtered_counts": {},
        "sample_hits": [],
        "result_count": 0,
        "error": None,
    }


def _log_step(log, step, **detail):
    entry = {"step": step}
    entry.update(detail)
    log["steps"].append(entry)


def format_search_log(log):
    """Human-readable search diagnostic log for UI and log files."""
    if not log:
        return ""

    lines = [
        f"Search log @ {log.get('timestamp', '')}",
        f"Query: {log.get('query', '')!r}",
    ]

    filters = log.get("filters") or {}
    if filters:
        lines.append("Filters:")
        for key, value in filters.items():
            if value is not None and value != "" and value is not False:
                lines.append(f"  - {key}: {value}")

    stats = log.get("index_stats") or {}
    if stats:
        lines.append("Index status:")
        for key in (
            "catalog_videos",
            "indexed_flags",
            "collection_exists",
            "segment_count",
            "qdrant_locked",
            "qdrant_dir",
        ):
            if key in stats:
                lines.append(f"  - {key}: {stats[key]}")

    for step in log.get("steps") or []:
        name = step.get("step", "step")
        detail_parts = [f"{k}={v}" for k, v in step.items() if k != "step"]
        suffix = f" ({', '.join(detail_parts)})" if detail_parts else ""
        lines.append(f"→ {name}{suffix}")

    lines.append(f"Qdrant raw hits: {log.get('qdrant_raw_hits', 0)}")

    filtered = log.get("filtered_counts") or {}
    if filtered:
        lines.append("Filtered out:")
        for reason, count in sorted(filtered.items()):
            if count:
                lines.append(f"  - {reason}: {count}")

    samples = log.get("sample_hits") or []
    if samples:
        lines.append("Sample hits (before/at filter stage):")
        for sample in samples:
            lines.append(
                f"  - {sample.get('filename', '?')} "
                f"semantic={sample.get('semantic_score', 0):.3f} "
                f"keyword={sample.get('keyword_score', 0):.3f} "
                f"combined={sample.get('combined_score', 0):.3f} "
                f"→ {sample.get('outcome', 'kept')}"
            )
            if sample.get("reject_reason"):
                lines.append(f"      reason: {sample['reject_reason']}")
            if sample.get("text_preview"):
                lines.append(f"      text: {sample['text_preview']}")

    lines.append(f"Results returned: {log.get('result_count', 0)}")
    if log.get("error"):
        lines.append(f"Error: {log['error']}")

    return "\n".join(lines)


def write_search_log(log):
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    text = format_search_log(log)
    with SEARCH_LOG_FILE.open("a", encoding="utf-8") as handle:
        handle.write(text)
        handle.write("\n" + ("=" * 72) + "\n")


def _finish_search(log, results, error):
    log["result_count"] = len(results or [])
    log["error"] = error
    write_search_log(log)
    return results, error, log


def search_videos(
    query,
    limit=20,
    folder_filter=None,
    date_from=None,
    date_to=None,
    min_size_mb=None,
    max_size_mb=None,
    min_score=None,
    extension_filter=None,
    db_file=None,
):
    log = _new_search_log(
        query,
        folder_filter=folder_filter,
        date_from=str(date_from) if date_from else None,
        date_to=str(date_to) if date_to else None,
        min_size_mb=min_size_mb,
        max_size_mb=max_size_mb,
        min_score=min_score,
        extension_filter=extension_filter,
        limit=limit,
    )

    if not query.strip():
        return _finish_search(log, [], "Enter a search query.")

    index_stats = get_search_index_stats(db_file)
    log["index_stats"] = index_stats
    _log_step(log, "index_stats_loaded", **index_stats)

    try:
        client = _open_qdrant_client_with_retry()
    except QdrantLockError as exc:
        msg = (
            "Search is briefly busy while a video is being indexed. "
            "Wait a few seconds and try again — search works between indexed files."
        )
        _log_step(log, "blocked", reason="qdrant_lock_error", detail=str(exc))
        return _finish_search(log, [], msg)

    if client is None:
        msg = "Search database not found. Run **Index search DB** on the Run Jobs tab first."
        _log_step(log, "blocked", reason="qdrant_dir_missing", path=str(QDRANT_DIR))
        return _finish_search(log, [], msg)

    try:
        if not client.collection_exists(COLLECTION_NAME):
            msg = "Search collection is empty. Run **Index Search DB** on the Run Jobs tab first."
            _log_step(log, "blocked", reason="collection_missing", collection=COLLECTION_NAME)
            return _finish_search(log, [], msg)

        catalog_paths = _catalog_paths(db_file)
        _log_step(
            log,
            "catalog_loaded",
            catalog_paths=len(catalog_paths),
            indexed_flags=index_stats.get("indexed_flags", 0),
        )

        embedder = _get_embedder()
        _log_step(log, "embedding_query", model=EMBEDDING_MODEL)
        vector = embedder.encode(query, show_progress_bar=False).tolist()
        _log_step(log, "embedding_ready", vector_dim=len(vector))

        min_size_bytes = int(min_size_mb * 1024 * 1024) if min_size_mb else None
        max_size_bytes = int(max_size_mb * 1024 * 1024) if max_size_mb else None

        qdrant_filter = _build_qdrant_filter(
            date_from=date_from,
            date_to=date_to,
            min_size_bytes=min_size_bytes,
            max_size_bytes=max_size_bytes,
        )
        _log_step(log, "qdrant_filter", applied=bool(qdrant_filter))

        has_post_filters = bool(
            folder_filter or date_from or date_to or min_size_mb or max_size_mb or extension_filter
        )
        query_limit = max(limit * 8, 40) if has_post_filters else max(limit * 3, 30)
        _log_step(
            log,
            "qdrant_query",
            query_limit=query_limit,
            has_post_filters=has_post_filters,
        )

        hits = client.query_points(
            collection_name=COLLECTION_NAME,
            query=vector,
            query_filter=qdrant_filter,
            limit=query_limit,
            with_payload=True,
        ).points
        log["qdrant_raw_hits"] = len(hits)
        _log_step(log, "qdrant_response", raw_hits=len(hits))

        results = []
        filtered_counts = {
            "stale_catalog": 0,
            "missing_file": 0,
            "folder_mismatch": 0,
            "extension_mismatch": 0,
            "below_min_score": 0,
        }
        sample_hits = []
        normalized_filter = folder_filter.rstrip("\\/") if folder_filter else None
        extension_filter = (extension_filter or "").strip().lower()
        if extension_filter and not extension_filter.startswith("."):
            extension_filter = f".{extension_filter}"
        min_score = float(min_score) if min_score is not None else 0.0
        _log_step(
            log,
            "post_filter_rules",
            min_score=min_score,
            folder_prefix=normalized_filter or "(all)",
            extension=extension_filter or "(any)",
        )

        def _record_sample(filename, semantic, overlap, combined, outcome, reject_reason="", text=""):
            if len(sample_hits) >= 8:
                return
            preview = (text or "").replace("\n", " ").strip()
            if len(preview) > 120:
                preview = preview[:117] + "..."
            sample_hits.append({
                "filename": filename,
                "semantic_score": round(semantic, 4),
                "keyword_score": round(overlap, 4),
                "combined_score": round(combined, 4),
                "outcome": outcome,
                "reject_reason": reject_reason,
                "text_preview": preview,
            })

        for hit in hits:
            payload = hit.payload or {}
            video_path = payload.get("video_path", "")
            text = payload.get("text", "")
            filename = payload.get("filename", Path(video_path).name)

            path_str = str(video_path)
            semantic = float(hit.score)
            overlap = _keyword_overlap(query, f"{text} {filename}")
            combined = (0.65 * semantic) + (0.35 * overlap)

            if catalog_paths and path_str not in catalog_paths:
                filtered_counts["stale_catalog"] += 1
                _record_sample(
                    filename, semantic, overlap, combined,
                    "rejected", "not in SQLite catalog", text,
                )
                continue
            if path_str and not Path(path_str).exists():
                filtered_counts["missing_file"] += 1
                _record_sample(
                    filename, semantic, overlap, combined,
                    "rejected", "video file missing on disk", text,
                )
                continue

            if normalized_filter and not path_str.startswith(normalized_filter):
                filtered_counts["folder_mismatch"] += 1
                _record_sample(
                    filename, semantic, overlap, combined,
                    "rejected", f"path does not start with {normalized_filter}", text,
                )
                continue

            if extension_filter and not path_str.lower().endswith(extension_filter):
                filtered_counts["extension_mismatch"] += 1
                _record_sample(
                    filename, semantic, overlap, combined,
                    "rejected", f"extension != {extension_filter}", text,
                )
                continue

            if combined < min_score:
                filtered_counts["below_min_score"] += 1
                _record_sample(
                    filename, semantic, overlap, combined,
                    "rejected", f"combined score {combined:.3f} < min {min_score:.3f}", text,
                )
                continue

            _record_sample(filename, semantic, overlap, combined, "kept", text=text)

            results.append({
                "score": round(combined, 4),
                "semantic_score": round(semantic, 4),
                "keyword_score": round(overlap, 4),
                "video_path": video_path,
                "filename": filename,
                "folder": payload.get("folder", ""),
                "modified_time": payload.get("modified_time", ""),
                "size_bytes": payload.get("size_bytes", 0),
                "start": float(payload.get("start", 0)),
                "end": float(payload.get("end", 0)),
                "start_label": format_timestamp(payload.get("start", 0)),
                "end_label": format_timestamp(payload.get("end", 0)),
                "text": text,
                "sources": payload.get("sources", []),
            })

        log["filtered_counts"] = filtered_counts
        log["sample_hits"] = sample_hits

        results.sort(key=lambda item: item["score"], reverse=True)

        if not results:
            if filtered_counts["stale_catalog"] or filtered_counts["missing_file"]:
                msg = (
                    "No matching segments in the current library. "
                    "Stale index entries were ignored — try **Reset search index** under Tools/System, "
                    "then run **Index search DB** again."
                )
                _log_step(log, "no_results", reason="stale_index_entries", **filtered_counts)
                return _finish_search(log, [], msg)
            msg = "No matching segments found."
            _log_step(log, "no_results", reason="all_filtered_or_no_match", **filtered_counts)
            return _finish_search(log, [], msg)

        _log_step(log, "success", returned=min(len(results), limit))
        return _finish_search(log, results[:limit], None)
    except Exception as exc:
        _log_step(log, "error", detail=str(exc))
        return _finish_search(log, [], f"Search failed: {exc}")
    finally:
        _close_qdrant_client(client)
