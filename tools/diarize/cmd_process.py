"""Process subcommand: whisperX transcription + pyannote diarization."""

import argparse
import json
import logging
import os
import subprocess
import sys
import tempfile
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from .config import Episode, PROJECT_ROOT, PYANNOTE_MODEL, WHISPER_MODEL
from .discovery import (
    discover_transcripts,
    find_tool,
    format_duration,
    match_episodes,
    scan_videos,
)

logger = logging.getLogger("whisperx_diarize")

# Worker globals (set per-process in worker_init)
_worker_whisper = None
_worker_align_model = None
_worker_align_meta = None
_worker_diarize = None
_worker_device = "cpu"


def worker_init(
    num_threads: int, hf_token: str, whisper_model: str, device: str = "cpu",
) -> None:
    """Load all three whisperX models in worker process."""
    global _worker_whisper, _worker_align_model  # noqa: PLW0603
    global _worker_align_meta, _worker_diarize, _worker_device  # noqa: PLW0603

    _worker_device = device

    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

    import torch
    torch.set_num_threads(num_threads)

    # PyTorch 2.8+ monkey-patch for pyannote model loading
    _original_load = torch.load

    def _patched_load(*args, **kwargs):
        kwargs["weights_only"] = False
        return _original_load(*args, **kwargs)

    torch.load = _patched_load
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

    compute_type = "float16" if device == "cuda" else "int8"

    import whisperx

    _worker_whisper = whisperx.load_model(
        whisper_model, device, compute_type=compute_type, language="en",
    )
    _worker_align_model, _worker_align_meta = whisperx.load_align_model(
        language_code="en", device=device,
    )
    from whisperx.diarize import DiarizationPipeline
    _worker_diarize = DiarizationPipeline(
        use_auth_token=hf_token, device=device,
    )


def _worker_init_fork(num_threads: int) -> None:
    """Lightweight init for forked workers (models inherited from parent via CoW)."""
    import torch
    torch.set_num_threads(num_threads)


def worker_process(episode_dict: dict, output_dir: str) -> dict:
    """Run whisperX pipeline on a single episode: transcribe + align + diarize."""
    import whisperx

    ep_id = episode_dict["episode_id"]
    video_path = Path(episode_dict["video_path"])
    output_path = Path(output_dir) / episode_dict["output_filename"]
    wav_path = None
    t0 = time.monotonic()

    try:
        # Extract audio
        tmp_dir = tempfile.mkdtemp(prefix="whisperx_")
        wav_path = Path(tmp_dir) / f"{ep_id}.wav"
        ffmpeg_cmd = ["ffmpeg", "-y", "-i", str(video_path)]
        audio_track = episode_dict.get("audio_track")
        if audio_track is not None:
            ffmpeg_cmd += ["-map", f"0:a:{audio_track}"]
        ffmpeg_cmd += ["-vn", "-acodec", "pcm_s16le",
                       "-ar", "16000", "-ac", "1", str(wav_path)]
        subprocess.run(ffmpeg_cmd, capture_output=True, check=True)
        wav_size_mb = wav_path.stat().st_size / (1024 * 1024)

        # Load audio
        audio = whisperx.load_audio(str(wav_path))

        # Step 1: Transcribe
        batch_size = 4 if _worker_device == "cuda" else 16
        result = _worker_whisper.transcribe(audio, batch_size=batch_size, language="en")

        # Step 2: Align (word-level timestamps)
        result = whisperx.align(
            result["segments"], _worker_align_model, _worker_align_meta,
            audio, _worker_device, return_char_alignments=False,
        )

        # Step 3: Diarize
        diarize_segments = _worker_diarize(audio)

        # Step 4: Assign speakers to words
        result = whisperx.assign_word_speakers(diarize_segments, result)

        # Build output segments
        segments = []
        speakers = set()
        for seg in result.get("segments", []):
            spk = seg.get("speaker", "")
            if spk:
                speakers.add(spk)
            words = []
            for w in seg.get("words", []):
                words.append({
                    "word": w.get("word", ""),
                    "start": round(w.get("start", 0.0), 3),
                    "end": round(w.get("end", 0.0), 3),
                    "speaker": w.get("speaker", ""),
                })
            segments.append({
                "start": round(seg.get("start", 0.0), 3),
                "end": round(seg.get("end", 0.0), 3),
                "text": seg.get("text", "").strip(),
                "speaker": spk,
                "words": words,
            })

        duration = segments[-1]["end"] if segments else 0.0
        elapsed = time.monotonic() - t0

        output = {
            "episode_id": ep_id,
            "title": episode_dict["title"],
            "series": episode_dict["series"],
            "season": episode_dict["season"],
            "episode": episode_dict["episode"],
            "transcript_path": episode_dict["transcript_path"],
            "video_path": str(video_path),
            "num_speakers": len(speakers),
            "duration_seconds": round(duration, 1),
            "processing_time_seconds": round(elapsed, 1),
            "models": {
                "whisper": episode_dict.get("whisper_model", WHISPER_MODEL),
                "diarize": PYANNOTE_MODEL,
            },
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "segments": segments,
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        return {
            "episode_id": ep_id, "success": True,
            "num_speakers": len(speakers), "duration": round(duration, 1),
            "elapsed": round(elapsed, 1), "wav_mb": round(wav_size_mb, 1),
            "n_segments": len(segments),
        }

    except Exception as e:
        elapsed = time.monotonic() - t0
        error_result = {
            "episode_id": ep_id, "error": str(e),
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(error_result, f, indent=2)
        except OSError:
            pass
        return {
            "episode_id": ep_id, "success": False,
            "error": str(e), "elapsed": round(elapsed, 1),
        }

    finally:
        if wav_path and wav_path.exists():
            wav_path.unlink(missing_ok=True)
        if wav_path:
            try:
                wav_path.parent.rmdir()
            except OSError:
                pass


def episode_to_dict(
    ep: Episode, whisper_model: str, audio_track: int | None = None,
) -> dict:
    d = {
        "episode_id": ep.episode_id,
        "output_filename": ep.output_filename,
        "title": ep.title,
        "series": ep.series,
        "season": ep.season,
        "episode": ep.episode,
        "transcript_path": str(ep.transcript_path),
        "video_path": str(ep.video_path),
        "whisper_model": whisper_model,
    }
    if audio_track is not None:
        d["audio_track"] = audio_track
    return d


def run_batch(
    episodes: list[Episode],
    output_dir: Path,
    workers: int,
    hf_token: str,
    whisper_model: str,
    device: str = "cpu",
    audio_track: int | None = None,
) -> None:
    total = len(episodes)
    if total == 0:
        logger.info("Nothing to process.")
        return

    threads_per_worker = max(1, 20 // workers)
    logger.info(
        "Starting %d worker(s) (%d threads each) for %d episodes",
        workers, threads_per_worker, total,
    )

    completed = 0
    succeeded = 0
    failed = 0
    failed_ids: list[str] = []
    batch_start = time.monotonic()

    # Single worker: run in-process to avoid fork+CUDA deadlock
    if workers == 1:
        worker_init(threads_per_worker, hf_token, whisper_model, device)
    else:
        # On Linux, use fork to share model memory via copy-on-write (CPU only)
        if sys.platform != "win32":
            worker_init(threads_per_worker, hf_token, whisper_model, device)
            import multiprocessing
            ctx = multiprocessing.get_context("fork")
            pool_kwargs = dict(
                max_workers=workers,
                mp_context=ctx,
                initializer=_worker_init_fork,
                initargs=(threads_per_worker,),
            )
        else:
            pool_kwargs = dict(
                max_workers=workers,
                initializer=worker_init,
                initargs=(threads_per_worker, hf_token, whisper_model, device),
            )
    def _log_result(ep: Episode, result: dict) -> None:
        nonlocal completed, succeeded, failed
        completed += 1
        pct = completed * 100 / total
        wall_elapsed = time.monotonic() - batch_start
        wall_per_ep = wall_elapsed / completed
        wall_remaining = (total - completed) * wall_per_ep

        if result["success"]:
            succeeded += 1
            logger.info(
                "[%03d/%03d %3.0f%%] %s %s -- %d speakers, %d segs, "
                "%.0fs audio, took %.0fs | elapsed %s, ETA %s",
                completed, total, pct,
                result["episode_id"], ep.title,
                result["num_speakers"], result["n_segments"],
                result["duration"], result["elapsed"],
                format_duration(wall_elapsed),
                format_duration(wall_remaining),
            )
        else:
            failed += 1
            failed_ids.append(result["episode_id"])
            logger.error(
                "[%03d/%03d %3.0f%%] %s FAILED: %s | elapsed %s",
                completed, total, pct,
                result["episode_id"], result.get("error", "unknown"),
                format_duration(wall_elapsed),
            )

    if workers == 1:
        for ep in episodes:
            ep_dict = episode_to_dict(ep, whisper_model, audio_track)
            result = worker_process(ep_dict, str(output_dir))
            # CPU fallback: retry OOM failures on CPU
            if (
                not result["success"]
                and device == "cuda"
                and "out of memory" in result.get("error", "").lower()
            ):
                import torch
                torch.cuda.empty_cache()
                logger.warning(
                    "%s: CUDA OOM, retrying on CPU...", result["episode_id"],
                )
                worker_init(threads_per_worker, hf_token, whisper_model, "cpu")
                result = worker_process(ep_dict, str(output_dir))
                # Restore CUDA models for next episode
                worker_init(threads_per_worker, hf_token, whisper_model, device)
            # Free CUDA memory between episodes to prevent fragmentation
            if device == "cuda":
                import torch
                torch.cuda.empty_cache()
            _log_result(ep, result)
    else:
        with ProcessPoolExecutor(**pool_kwargs) as pool:
            futures = {
                pool.submit(
                    worker_process,
                    episode_to_dict(ep, whisper_model, audio_track),
                    str(output_dir),
                ): ep
                for ep in episodes
            }
            for future in as_completed(futures):
                _log_result(futures[future], future.result())

    wall_time = time.monotonic() - batch_start
    logger.info("")
    logger.info("=" * 60)
    logger.info("Batch complete in %s", format_duration(wall_time))
    logger.info("  Succeeded: %d", succeeded)
    logger.info("  Failed:    %d", failed)
    if failed_ids:
        logger.info("  Failed episodes: %s", ", ".join(failed_ids))


def cmd_process(args: argparse.Namespace) -> None:
    find_tool("ffmpeg")

    # Resolve device
    device = getattr(args, "device", "auto")
    if device == "auto":
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"

    hf_token = os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_TOKEN", "")
    if not hf_token and not args.dry_run:
        print("Error: HUGGINGFACE_TOKEN or HF_TOKEN env var not set", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not args.dry_run:
        from .cmd_status import setup_logging
        setup_logging(output_dir)

    all_episodes = discover_transcripts(PROJECT_ROOT)
    logger.info("Found %d transcripts", len(all_episodes))

    logger.info("Scanning video directories: %s",
                ", ".join(str(d) for d in args.video_dirs))
    video_index = scan_videos(args.video_dirs)
    logger.info("Found %d video files", len(video_index))

    matched, unmatched = match_episodes(all_episodes, video_index)

    # Apply filters
    if args.series != "all":
        code = args.series.upper()
        matched = [e for e in matched if e.series == code]
        unmatched = [e for e in unmatched if e.series == code]
    if args.season is not None:
        matched = [e for e in matched if e.season == args.season]
        unmatched = [e for e in unmatched if e.season == args.season]
    if args.episode:
        matched = [e for e in matched if e.episode_id in args.episode]
        unmatched = []

    # Skip already-processed
    skipped = []
    if not args.force:
        remaining = []
        for ep in matched:
            out_path = output_dir / ep.output_filename
            if out_path.exists():
                try:
                    with open(out_path) as f:
                        data = json.load(f)
                    if "error" in data and "segments" not in data:
                        remaining.append(ep)
                        continue
                except (json.JSONDecodeError, OSError):
                    remaining.append(ep)
                    continue
                skipped.append(ep)
            else:
                remaining.append(ep)
        matched = remaining

    if args.dry_run:
        print(f"\n{'=' * 80}")
        print(f"Matched: {len(matched)} | Unmatched: {len(unmatched)} | "
              f"Already done: {len(skipped)}")
        print(f"{'=' * 80}\n")
        if matched:
            print("WILL PROCESS:")
            for ep in matched:
                print(f"  {ep.episode_id:12s} {ep.title}")
                print(f"    -> {ep.video_path}")
        if skipped:
            print("\nALREADY PROCESSED (use --force to re-run):")
            for ep in skipped:
                print(f"  {ep.episode_id:12s} {ep.title}")
        if unmatched:
            print("\nNO VIDEO FOUND:")
            for ep in unmatched:
                print(f"  {ep.episode_id:12s} {ep.title}")
        return

    if unmatched:
        logger.info("No video found for %d transcripts:", len(unmatched))
        for ep in unmatched:
            logger.info("  %s %s", ep.episode_id, ep.title)
    if skipped:
        logger.info("Skipping %d already-processed episodes", len(skipped))

    audio_track = getattr(args, "audio_track", None)
    logger.info("Processing %d episodes with %d workers (device=%s)",
                len(matched), args.workers, device)
    if device == "cuda" and args.workers > 1:
        logger.warning("Multiple GPU workers compete for VRAM; consider --workers 1")
    run_batch(matched, output_dir, args.workers, hf_token, args.whisper_model,
              device, audio_track=audio_track)
