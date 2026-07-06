import argparse
import datetime
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from job_utils import (
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    QDRANT_DIR,
    add_processing_args,
    find_videos,
    metadata_json_path,
    overwrite_from_args,
    read_status,
    should_stop,
    update_video_flag,
    write_status,
)
from ml_bootstrap import prepare_ml_environment
from pipeline_utils import (
    add_pipeline_control_args,
    clear_step_failure,
    filter_videos_for_step,
    get_video_row,
    load_video_allowlist,
    record_step_failure,
    skip_mode_from_args,
    step_overwrite_from_args,
    step_status,
)


def point_id(video_path, start, end):
    key = f"{video_path}|{start:.3f}|{end:.3f}"
    return int(hashlib.md5(key.encode("utf-8")).hexdigest()[:16], 16)


def load_embedder():
    try:
        prepare_ml_environment()
        from sentence_transformers import SentenceTransformer
    except Exception as e:
        raise RuntimeError(
            "Could not load sentence-transformers embedding model. "
            "Run Install / Update AI Dependencies from the Tools/System tab. "
            f"({e})"
        ) from e

    print(f"Loading embedding model: {EMBEDDING_MODEL}", flush=True)
    return SentenceTransformer(EMBEDDING_MODEL)


def get_qdrant_client(vector_size):
    try:
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams
    except Exception as e:
        raise RuntimeError(
            f"Missing qdrant-client. Run Install / Update AI Dependencies. ({e})"
        ) from e

    QDRANT_DIR.mkdir(parents=True, exist_ok=True)
    client = QdrantClient(path=str(QDRANT_DIR))

    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )
        print(f"Created Qdrant collection: {COLLECTION_NAME}", flush=True)

    return client


def delete_video_points(client, video_path):
    from qdrant_client.models import Filter, FieldCondition, MatchValue

    client.delete(
        collection_name=COLLECTION_NAME,
        points_selector=Filter(
            must=[
                FieldCondition(
                    key="video_path",
                    match=MatchValue(value=str(video_path)),
                )
            ]
        ),
    )


def modified_ts_from_metadata(metadata):
    value = metadata.get("modified_time", "")
    if not value:
        return 0.0
    try:
        return datetime.datetime.fromisoformat(value).timestamp()
    except ValueError:
        return 0.0


def index_video(client, embedder, metadata, overwrite):
    video_path = metadata.get("video", "")
    chunks = metadata.get("search_chunks", [])

    if not chunks:
        print(f"No search chunks for: {video_path}", flush=True)
        return 0

    delete_video_points(client, video_path)

    from qdrant_client.models import PointStruct

    texts = [chunk["text"] for chunk in chunks]
    vectors = embedder.encode(texts, show_progress_bar=False)

    points = []
    for chunk, vector in zip(chunks, vectors):
        start = float(chunk.get("start", 0))
        end = float(chunk.get("end", start))
        points.append(
            PointStruct(
                id=point_id(video_path, start, end),
                vector=vector.tolist(),
                payload={
                    "video_path": video_path,
                    "filename": metadata.get("filename", Path(video_path).name),
                    "folder": metadata.get("folder", str(Path(video_path).parent)),
                    "modified_time": metadata.get("modified_time", ""),
                    "modified_ts": modified_ts_from_metadata(metadata),
                    "size_bytes": int(metadata.get("size_bytes", 0) or 0),
                    "start": start,
                    "end": end,
                    "text": chunk.get("text", ""),
                    "sources": chunk.get("sources", []),
                },
            )
        )

    client.upsert(collection_name=COLLECTION_NAME, points=points)
    return len(points)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", required=True)
    parser.add_argument("--db", required=True)
    parser.add_argument("--status-file", required=True)
    add_processing_args(parser)
    add_pipeline_control_args(parser)
    args = parser.parse_args()

    folder = Path(args.folder)
    status_file = Path(args.status_file)
    global_overwrite = overwrite_from_args(args)
    overwrite = step_overwrite_from_args(args, "index", global_overwrite)
    skip_mode = skip_mode_from_args(args)
    allowlist = load_video_allowlist(args)

    status = read_status(status_file)
    status.update({
        "status": "running",
        "current": "Preparing Qdrant index...",
        "percent": 0,
        "processed": 0,
        "total": 0,
    })
    write_status(status_file, status)

    print(f"Qdrant indexing started for: {folder}", flush=True)

    videos = filter_videos_for_step(
        find_videos(folder),
        args.db,
        "index",
        skip_mode,
        overwrite,
        allowlist,
    )
    total = len(videos)

    status.update({
        "total": total,
        "current": f"Found {total} video files.",
    })
    write_status(status_file, status)
    print(f"Found {total} video files.", flush=True)

    client = None
    try:
        embedder = load_embedder()
        if hasattr(embedder, "get_embedding_dimension"):
            vector_size = embedder.get_embedding_dimension()
        else:
            vector_size = embedder.get_sentence_embedding_dimension()
        client = get_qdrant_client(vector_size)
    except Exception as e:
        status.update({
            "status": "failed",
            "current": str(e),
            "finished_at": datetime.datetime.now().isoformat(timespec="seconds"),
        })
        write_status(status_file, status)
        print(f"FAILED: {e}", flush=True)
        sys.exit(1)

    indexed_count = 0

    try:
        for i, video in enumerate(videos, start=1):
            if should_stop(status_file):
                status.update({
                    "status": "stopped",
                    "current": "Stopped by user.",
                    "finished_at": datetime.datetime.now().isoformat(timespec="seconds"),
                })
                write_status(status_file, status)
                print("Stopped by user.", flush=True)
                sys.exit(0)

            metadata_file = metadata_json_path(video)

            status.update({
                "current": f"Indexing: {video}",
                "processed": i - 1,
                "total": total,
                "percent": int(((i - 1) / total) * 100) if total else 100,
            })
            write_status(status_file, status)

            if not metadata_file.exists():
                print(f"Skipping (no metadata): {video}", flush=True)
            else:
                row = get_video_row(args.db, video)
                if not overwrite and step_status(row, "index") == "complete":
                    print(f"Skipping (already indexed): {video}", flush=True)
                else:
                    print(f"\nIndexing: {video}", flush=True)
                    try:
                        metadata = json.loads(
                            metadata_file.read_text(encoding="utf-8")
                        )
                        chunk_count = index_video(
                            client, embedder, metadata, overwrite=overwrite
                        )
                        if chunk_count:
                            update_video_flag(args.db, video, "indexed_in_qdrant")
                            clear_step_failure(video, "index")
                            indexed_count += chunk_count
                            print(f"Indexed {chunk_count} segments", flush=True)
                    except Exception as e:
                        record_step_failure(video, "index", str(e))
                        print(f"FAILED: {video}", flush=True)
                        print(str(e), flush=True)

            percent = int((i / total) * 100) if total else 100
            status.update({
                "processed": i,
                "total": total,
                "percent": percent,
                "current": str(video),
            })
            write_status(status_file, status)

        status.update({
            "status": "complete",
            "percent": 100,
            "current": f"Qdrant indexing complete. Indexed {indexed_count} segments.",
            "finished_at": datetime.datetime.now().isoformat(timespec="seconds"),
        })
        write_status(status_file, status)
        print(f"Qdrant indexing complete. Indexed {indexed_count} segments.", flush=True)
    finally:
        if client is not None:
            try:
                client.close()
            except Exception:
                pass


if __name__ == "__main__":
    main()
