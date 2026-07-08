# Changelog

All notable changes to AI Video Indexer are documented here.

## [0.2.2] — 2026-07-08

### Added

- **Overnight / scheduled runs** — Tools/System schedule with weekdays, start/stop times, auto-resume, and Windows Task Scheduler integration (`register_schedule_task.bat`). Only scheduler-started jobs are auto-stopped; manual Run Jobs are unaffected.
- **Search while indexing** — Qdrant lock is released between indexed videos; search retries briefly instead of blocking for the whole job.
- **Batch SQLite in pipeline** — catalog writes reuse one connection and commit in batches; scan upserts merge sidecar sync into a single transaction; pipeline steps preload row caches to cut per-video reads.
- **HEVC transcode (standalone)** — Run Jobs section to re-encode to H.265 with quality presets, NVENC/x265 encoder choice, audio options, resolution cap, suffix or replace output modes.
- **Disk space visibility** — Library tab shows volume free/total (local and many UNC shares); HEVC transcode checks space before starting.
- **HEVC target bitrate / file size** — alternative to CRF for predictable output size.
- **Pipeline failures panel** — Tools/System lists per-video errors from pipeline, normalize, and transcode jobs with a clear-all option.

### Changed

- **Tab layout** — single app-level auto-refresh when jobs or schedule are active; removed tab fragments that could collapse the UI.
- **Streamlit layout API** — `use_container_width` updated to `width="stretch"` where supported.

### Fixed

- **Logs → Clear All Logs** — clears only `logs/*.log` files; job history on Run Jobs is preserved.
- **Error handling** — hardened config load, job crash handlers, clearer Dashboard/job messages, safer search and job startup paths.
- **False “started” toasts** — pipeline/transcode only reports success when the job process actually launches.
- **`complete_with_failures` status** — partial pipeline success is now visible on Dashboard and job history.
- **Normalize & transcode failures** — recorded in `pipeline_failures.json` and shown in the failures panel (not only CSV logs).

## [0.2.1] — 2026-07-07

### Added

- **Single-file pipeline** — select one or more videos on the Library tab or Run Jobs and run the pipeline on just those files.
- **Duplicate detection** — scan stores duration and content fingerprint; Library and Tools/System show duplicate groups.
- **Search debug log** — Search tab expander plus `logs/search.log` with index stats, Qdrant hit counts, filter breakdown, and sample reject reasons.
- **Search export** — download results as CSV or Markdown after a query.
- **Person tags** — metadata extracts people-related tags from vision/transcript text for search (description-based, not face recognition).
- **Thumbnails** — `.thumbnail.jpg` sidecar on scan; preview column in Library; frame-at-hit preview in Search results.
- **Whisper language in Library** — auto-detected per-file language shown in the catalog table.
- **Resume step mismatch warning** — explains when pipeline step checkboxes changed since a stopped run (resume restarts from the beginning).
- **Vision dependency check** — Run Jobs warns and offers **Install vision dependencies** when the vision stack is missing.
- **Live install progress** — Install / Update AI Dependencies shows step progress on Dashboard and Tools/System (auto-refreshing).

### Changed

- **`start.bat`** — opens the default browser to `http://localhost:8501` after launch.
- **Logs tab** — removed broken auto-refresh fragment that could collapse all tabs; optional auto-refresh checkbox instead.
- **Install scripts** — preserve job metadata (`pid`, `log_file`, etc.) so Dashboard tracking works for dependency installs.

### Fixed

- **White screen on Run Jobs** — `find_vision_resume_mismatches` no longer passes strings where `Path` objects are required; failures show a caption instead of crashing the app.
- **Install progress invisible on Dashboard** — dependency install no longer wipes the job status file and get marked failed immediately.
- **Search troubleshooting** — debug log makes “no results” diagnosable (empty index, filters, stale paths, min score, etc.).

## [0.2] — 2026-07-06

### Added

- **Pipeline time estimator** on Run Jobs — rough duration ranges with/without vision, based on library size, GPU tier, Whisper model, vision model, and frame interval.
- **Vision model selection** — Qwen2.5-VL 3B (default), Qwen2.5-VL 7B, and Qwen2-VL 2B; wired through UI, pipeline, and sidecar metadata.
- **Vision frame resume** — interrupted vision runs save progress incrementally and resume from the last completed frame within a video.
- **End-of-pipeline library scan** — optional rescan after processing completes (default on); updates catalog flags and Library tab counts.
- **Pipeline resume after crash or reboot** — failed/stopped runs leave resume state; stale jobs are detected and marked resumable instead of discarding progress.
- **Quick pipeline actions** — Index missing only, Refresh stale metadata/index, Retry incomplete videos.
- **Discord partial-success notifications** — pipelines that finish with per-video failures report *finished with failures* instead of plain *complete*.
- **Model help** — popovers on Transcription/Vision settings plus expanded Whisper and Vision model sections in Help.
- **Vision settings mismatch warning** — UI warns when resuming a pipeline if vision model or frame settings changed; incompatible partial vision files restart from scratch.

### Changed

- **Job progress** shows elapsed wall time instead of ETA; Dashboard and logs auto-refresh every second.
- **Dashboard GPU display** uses a cleaner two-column layout.
- **Tools / Status** tab renamed to **Tools/System**.
- **Processing Settings** grouped into General, Transcription, and Vision (side-by-side).
- **Search** — folder filter always visible; other filters moved under *More search filters*.
- **Run Jobs layout** — step checkboxes, scope, and resume sit directly above **Run Selected Pipeline**; quick actions moved below the main button.
- **Scan library** is manual only (Run Jobs → Scan Library); no longer a pre-pipeline checkbox.
- **Discord webhook** is read from `config.json` at notify time — not stored in job JSON files.
- **Normalize dry-run** removed from UI and pipeline.
- **`setup.bat`** — improved install flow and dependency handling.
- **`requirements.txt`** — updated package pins for vision and transcription stack.

### Fixed

- **Search engine job lock detection** — corrected jobs directory path so Qdrant indexing waits for active pipeline jobs.
- **Middle progress bar** — only shown when the current step reports a non-zero total.
- **Install dependencies button** — correctly re-enables after stale running jobs are cleaned up.
- **Release packaging** — `pipeline_estimate.py` included in `package_release.ps1`.
- **Vision `has_vision` flag** — only set when vision sidecar is fully complete, not mid-run.

## [0.1] — 2026-07-05

Initial release — Streamlit app for scanning home videos, running a local AI pipeline (transcribe, vision, metadata, Qdrant search), and searching by speech and scene content with timestamps.
