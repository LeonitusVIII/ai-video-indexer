# Changelog

All notable changes to AI Video Indexer are documented here.

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
