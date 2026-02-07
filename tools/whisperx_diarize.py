#!/usr/bin/env python3
"""Combined whisperX diarization + validation for Adventure Time transcripts.

Uses whisperX to get who + what + when in a single pass:
  - faster-whisper for transcription
  - wav2vec2 for word-level alignment
  - pyannote for speaker diarization

Also supports SpeechBrain ECAPA-TDNN speaker embeddings for unsupervised
character identification (building per-cluster voice profiles, then labeling
clusters via show knowledge).

Subcommands:
    process         Run whisperX pipeline on video files, save enriched JSON
    validate        Map speaker clusters to characters, validate/fix transcripts
    embed-clusters  Build per-cluster embeddings from anonymous speakers
    embed-label     Save cluster→character mapping and merge into profiles

Usage:
    python tools/whisperx_diarize.py process --episode AT.S09E11
    python tools/whisperx_diarize.py validate --episode AT.S09E11 --write
    python tools/whisperx_diarize.py embed-clusters --episode AT.S08E08
    python tools/whisperx_diarize.py embed-label --episode AT.S08E08 --map SPEAKER_00=Finn ...
"""

import argparse
import io
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import warnings
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EPISODE_RE = re.compile(r"S(\d{2})E(\d{2,3})")
MULTI_EP_RE = re.compile(r"S(\d{2})E(\d{2,3})-E?(\d{2,3})")
SPEAKER_RE = re.compile(r"^([A-Z][A-Za-z \u2019'.\-()]+):\s")
PLACEHOLDER_RE = re.compile(r"^\?\?\?:\s+|^([A-Z][A-Za-z .\-]+) \[\?\]:\s+")
_NORM_RE = re.compile(r"[^\w\s']")
_SPACE_RE = re.compile(r"\s+")

FC_PATTERN = re.compile(
    r"Fionna[. ]and[. ]Cake[. ]S\d|Fionna and Cake[/\\]Season"
)

PYANNOTE_MODEL = "pyannote/speaker-diarization-3.1"
WHISPER_MODEL = "medium.en"
if sys.platform == "win32":
    DEFAULT_VIDEO_DIRS = [Path("D:/Shows"), Path("S:/Shows")]
elif Path("/shows-d").is_dir():
    # Docker container with mounted volumes
    DEFAULT_VIDEO_DIRS = [Path("/shows-d"), Path("/shows-s")]
else:
    # WSL2 with Windows drives mounted
    DEFAULT_VIDEO_DIRS = [Path("/mnt/d/Shows"), Path("/mnt/s/Shows")]

TRANSCRIPT_SERIES_DIRS = {
    "AT": "Adventure Time",
    "DL": "Adventure Time Distant Lands",
    "FC": "Adventure Time Fionna and Cake",
}

MIN_MATCH_RATIO = 0.55
MIN_FIX_CONFIDENCE = 0.6
MIN_CLUSTER_VOTES = 3
MULTI_SPEAKER = frozenset({"both", "all", "everyone", "together"})

# Embedding constants
ECAPA_SOURCE = "speechbrain/spkrec-ecapa-voxceleb"
ECAPA_SAVEDIR = "diarization/models/ecapa"
EMBED_DIM = 192
MIN_EMBED_DURATION = 1.5
MAX_EMBED_DURATION = 15.0
SEASON_BUCKETS: dict[str, list[tuple[int, int]]] = {
    "Finn": [(1, 3), (4, 6), (7, 10)],
}
BUCKET_RE = re.compile(r"(.+)_S(\d{2})-S(\d{2})$")

# Character alias resolution (from the-enchiridion sibling project)
ENCHIRIDION_CHARACTERS = (
    Path(__file__).resolve().parent.parent.parent
    / "the-enchiridion" / "src" / "data" / "characters.json"
)

# Preferred canonical transcript name per character id
CANONICAL_SPEAKER: dict[str, str] = {
    "finn": "Finn",
    "jake": "Jake",
    "bmo": "BMO",
    "princess-bubblegum": "Princess Bubblegum",
    "marceline": "Marceline",
    "ice-king": "Ice King",
    "lumpy-space-princess": "LSP",
    "lady-rainicorn": "Lady Rainicorn",
    "flame-princess": "Flame Princess",
    "hunson-abadeer": "Hunson Abadeer",
    "patience-st-pim": "Patience St. Pim",
    "peppermint-butler": "Peppermint Butler",
    "lemongrab": "Lemongrab",
    "betty-grof": "Betty",
    "gunter": "Gunter",
    "fern": "Fern",
    "susan-strong": "Susan Strong",
    "magic-man": "Magic Man",
    "tree-trunks": "Tree Trunks",
    "cinnamon-bun": "Cinnamon Bun",
    "banana-man": "Banana Man",
    "martin-mertens": "Martin",
    "fionna": "Fionna",
    "cake": "Cake",
    "prince-gumball": "Prince Gumball",
    "marshall-lee": "Marshall Lee",
}

# Aliases that should NOT be merged (different voice / performance)
VOICE_SEPARATE: frozenset[str] = frozenset({
    "Simon", "Simon Petrikov",          # Tom Kenny normal voice ≠ Ice King
    "Normal Man", "King Man",           # Magic Man reformed
    "Dirt Beer Guy",                    # Different from Root Beer Guy
    "Fern", "Grass Finn", "Fern the Human",  # Different from Finn
    "Ice Thing",                        # Different from Gunter
    "Sweet P", "Sweet Pig-Trunks",      # Different from The Lich
    "Punch Bowl",                       # Different from Uncle Gumbald
    "Manfried",                         # Different from Aunt Lolly
    "Crunchy",                          # Different from Cousin Chicle
    "Nectr",                            # Different from Lemongrab
    "Winter King",                      # Different from Ice King (F&C)
})

# Manual transcript abbreviations not in characters.json aliases
MANUAL_ALIASES: dict[str, str] = {
    "Patience": "Patience St. Pim",
    "Hunson": "Hunson Abadeer",
    "Pep But": "Peppermint Butler",
    "Pep-But": "Peppermint Butler",
    "Simon Petrikov": "Simon",  # Same voice, separate from Ice King
}

PROGRESS_FILE = "validation_progress.json"

logger = logging.getLogger("whisperx_diarize")

# Worker globals (set per-process in worker_init)
_worker_whisper = None
_worker_align_model = None
_worker_align_meta = None
_worker_diarize = None
_worker_device = "cpu"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class Episode:
    series: str
    season: int
    episode: int
    title: str
    transcript_path: Path
    video_path: Path | None = None

    @property
    def episode_id(self) -> str:
        return f"{self.series}.S{self.season:02d}E{self.episode:02d}"

    @property
    def output_filename(self) -> str:
        return f"{self.episode_id}.json"


@dataclass
class TLine:
    line_num: int
    raw: str
    speaker: str
    text: str
    is_scene: bool
    is_placeholder: bool
    w_start: float = -1.0
    w_end: float = -1.0
    w_ratio: float = 0.0
    dia_speaker: str = ""
    inferred: str = ""
    validation: str = ""


@dataclass
class ClusterMap:
    cluster: str
    character: str
    votes: int
    total: int

    @property
    def confidence(self) -> float:
        return self.votes / max(self.total, 1)


@dataclass
class EpResult:
    episode_id: str = ""
    title: str = ""
    total: int = 0
    labeled: int = 0
    unlabeled: int = 0
    matched: int = 0
    n_clusters: int = 0
    agree: int = 0
    disagree: int = 0
    fixed: int = 0
    unknown: int = 0
    disagree_details: list = field(default_factory=list)
    cluster_info: dict = field(default_factory=dict)
    error: str = ""


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def find_tool(name: str) -> str:
    path = shutil.which(name)
    if not path:
        print(f"Error: {name} not found on PATH", file=sys.stderr)
        sys.exit(1)
    return path


def classify_series(path_str: str) -> str | None:
    if FC_PATTERN.search(path_str):
        return "FC"
    if "Distant Lands" in path_str or "Distant.Lands" in path_str:
        return "DL"
    if "Adventure Time" in path_str or "Adventure.Time" in path_str:
        return "AT"
    return None


def parse_episode_code(filename: str) -> tuple[int, int] | None:
    m = EPISODE_RE.search(filename)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None


def extract_title(filename: str) -> str:
    stem = Path(filename).stem
    m = EPISODE_RE.search(stem)
    if m:
        after = stem[m.end():]
        if after.startswith("."):
            after = after[1:]
        return after.replace(".", " ")
    return stem


def format_duration(seconds: float) -> str:
    if seconds >= 3600:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        return f"{h}h {m}m"
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m}m {s}s"


def _normalize(s: str) -> str:
    return _SPACE_RE.sub(" ", _NORM_RE.sub(" ", s.lower())).strip()


def _word_set(s: str) -> set[str]:
    return set(s.split())


def _word_overlap(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / max(len(a | b), 1)


def _best_score(a: str, b: str) -> float:
    sm = SequenceMatcher(None, a, b)
    ratio = sm.ratio()
    shorter = min(len(a), len(b))
    if shorter > 0:
        matched_chars = sum(block.size for block in sm.get_matching_blocks())
        containment = matched_chars / shorter
        return max(ratio, containment)
    return ratio


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def discover_transcripts(root: Path) -> list[Episode]:
    episodes = []
    for series_code, series_dir in TRANSCRIPT_SERIES_DIRS.items():
        series_path = root / series_dir
        if not series_path.is_dir():
            continue
        for txt_path in sorted(series_path.rglob("*.txt")):
            code = parse_episode_code(txt_path.name)
            if not code:
                continue
            season, ep = code
            title = extract_title(txt_path.name)
            episodes.append(Episode(
                series=series_code, season=season, episode=ep,
                title=title, transcript_path=txt_path,
            ))
    return episodes


def scan_videos(roots: list[Path]) -> dict[tuple[str, int, int], Path]:
    index: dict[tuple[str, int, int], Path] = {}
    for root in roots:
        if not root.is_dir():
            logger.warning("Video directory not found: %s", root)
            continue
        for mkv in root.rglob("*.mkv"):
            series = classify_series(str(mkv))
            if series is None:
                continue
            multi = MULTI_EP_RE.search(mkv.name)
            if multi:
                season = int(multi.group(1))
                ep_start = int(multi.group(2))
                ep_end = int(multi.group(3))
                for ep in range(ep_start, ep_end + 1):
                    key = (series, season, ep)
                    if key not in index:
                        index[key] = mkv
                continue
            code = parse_episode_code(mkv.name)
            if not code:
                continue
            season, ep = code
            key = (series, season, ep)
            if key not in index:
                index[key] = mkv
    return index


def match_episodes(
    episodes: list[Episode],
    video_index: dict[tuple[str, int, int], Path],
) -> tuple[list[Episode], list[Episode]]:
    matched, unmatched = [], []
    for ep in episodes:
        key = (ep.series, ep.season, ep.episode)
        video = video_index.get(key)
        if video:
            ep.video_path = video
            matched.append(ep)
        else:
            unmatched.append(ep)
    return matched, unmatched


# ---------------------------------------------------------------------------
# Process: worker functions
# ---------------------------------------------------------------------------


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
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(video_path), "-vn", "-acodec", "pcm_s16le",
             "-ar", "16000", "-ac", "1", str(wav_path)],
            capture_output=True, check=True,
        )
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


def episode_to_dict(ep: Episode, whisper_model: str) -> dict:
    return {
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


# ---------------------------------------------------------------------------
# Process: batch orchestration
# ---------------------------------------------------------------------------


def run_batch(
    episodes: list[Episode],
    output_dir: Path,
    workers: int,
    hf_token: str,
    whisper_model: str,
    device: str = "cpu",
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
            ep_dict = episode_to_dict(ep, whisper_model)
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
                    worker_process, episode_to_dict(ep, whisper_model),
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


# ---------------------------------------------------------------------------
# Validate: transcript parsing + matching
# ---------------------------------------------------------------------------


def parse_transcript(text: str) -> list[TLine]:
    lines = []
    for i, raw in enumerate(text.split("\n"), 1):
        stripped = raw.strip()
        if not stripped:
            continue
        is_scene = stripped.startswith("[")
        is_ph = False
        speaker = ""
        dialogue = stripped

        m = SPEAKER_RE.match(stripped)
        if m:
            speaker = m.group(1).strip()
            dialogue = stripped[m.end():].strip()
        else:
            pm = PLACEHOLDER_RE.match(stripped)
            if pm:
                is_ph = True
                dialogue = stripped[pm.end():].strip()

        lines.append(TLine(
            line_num=i, raw=stripped, speaker=speaker,
            text=dialogue, is_scene=is_scene, is_placeholder=is_ph,
        ))
    return lines


def match_to_segments(lines: list[TLine], segments: list[dict]) -> None:
    """Match transcript lines to whisperX segments.  Mutates lines in-place.

    Each segment already has text + speaker, so we fuzzy-match transcript text
    to segment text and read the speaker directly.  No separate diarization
    lookup needed.
    """
    seg_idx = 0
    seg_norm = [_normalize(s.get("text", "")) for s in segments]
    seg_words = [_word_set(n) for n in seg_norm]

    for tl in lines:
        if tl.is_scene or not tl.text:
            continue
        text = _normalize(tl.text)
        if not text:
            continue
        tl_words = _word_set(text)
        best_r = 0.0
        best_spk = ""
        best_s, best_e = -1.0, -1.0

        lo = max(0, seg_idx - 5)
        hi = min(len(segments), seg_idx + 25)

        for j in range(lo, hi):
            seg = segments[j]

            if _word_overlap(tl_words, seg_words[j]) > 0.15:
                r = _best_score(text, seg_norm[j])
                if r > best_r:
                    best_r = r
                    best_spk = seg.get("speaker", "")
                    best_s, best_e = seg["start"], seg["end"]

            # Concatenate adjacent segments (transcript line may span multiple)
            cat_words = set(seg_words[j])
            cat = seg_norm[j]
            cat_e = seg["end"]
            for k in range(j + 1, min(j + 4, len(segments))):
                cat += " " + seg_norm[k]
                cat_e = segments[k]["end"]
                cat_words |= seg_words[k]
                if _word_overlap(tl_words, cat_words) > 0.15:
                    cr = _best_score(text, cat)
                    if cr > best_r:
                        best_r = cr
                        best_spk = seg.get("speaker", "")
                        best_s = seg["start"]
                        best_e = cat_e

            if best_r >= 0.85:
                break

        # Wider fallback
        if best_r < MIN_MATCH_RATIO and len(text) > 10:
            for j in range(len(segments)):
                if lo <= j < hi:
                    continue
                if _word_overlap(tl_words, seg_words[j]) < 0.2:
                    continue
                r = _best_score(text, seg_norm[j])
                if r > best_r:
                    best_r = r
                    best_spk = segments[j].get("speaker", "")
                    best_s = segments[j]["start"]
                    best_e = segments[j]["end"]
                    if r >= 0.85:
                        break

        if best_r >= MIN_MATCH_RATIO:
            tl.w_start, tl.w_end, tl.w_ratio = best_s, best_e, best_r
            tl.dia_speaker = best_spk
            for j in range(seg_idx, len(segments)):
                if segments[j]["start"] >= best_e:
                    seg_idx = j
                    break


def build_cluster_map(lines: list[TLine]) -> dict[str, ClusterMap]:
    votes: dict[str, Counter] = defaultdict(Counter)
    for tl in lines:
        if not tl.speaker or not tl.dia_speaker:
            continue
        if tl.speaker.lower() in MULTI_SPEAKER:
            continue
        votes[tl.dia_speaker][tl.speaker] += 1

    mapping = {}
    for cluster, ctr in votes.items():
        top, n = ctr.most_common(1)[0]
        total = sum(ctr.values())
        mapping[cluster] = ClusterMap(cluster, top, n, total)
    return mapping


def build_merged_character_map(
    cmap: dict[str, ClusterMap],
) -> dict[str, tuple[int, int, float, set[str]]]:
    """Merge clusters that mapped to the same character.

    Returns {character: (total_votes, total_lines, confidence, cluster_ids)}.
    """
    char_cms: dict[str, list[ClusterMap]] = defaultdict(list)
    for cm in cmap.values():
        char_cms[cm.character].append(cm)
    merged = {}
    for char, cms in char_cms.items():
        total_votes = sum(cm.votes for cm in cms)
        total_lines = sum(cm.total for cm in cms)
        merged[char] = (
            total_votes, total_lines,
            total_votes / max(total_lines, 1),
            {cm.cluster for cm in cms},
        )
    return merged


def _context_speaker(lines: list[TLine], idx: int, window: int = 15) -> str:
    """Return speaker if nearest labeled lines on both sides agree."""
    prev = ""
    for j in range(idx - 1, max(idx - window, -1), -1):
        if lines[j].is_scene or not lines[j].text:
            continue
        if lines[j].speaker:
            prev = lines[j].speaker
            break
    nxt = ""
    for j in range(idx + 1, min(idx + window, len(lines))):
        if lines[j].is_scene or not lines[j].text:
            continue
        if lines[j].speaker:
            nxt = lines[j].speaker
            break
    if prev and nxt and prev == nxt:
        return prev
    return ""


def _same_cluster_neighbor(lines: list[TLine], idx: int, window: int = 6) -> str:
    """Return speaker of nearest labeled line sharing the same diarization cluster."""
    tl = lines[idx]
    if not tl.dia_speaker:
        return ""
    for j in range(idx - 1, max(idx - window, -1), -1):
        if lines[j].is_scene or not lines[j].text:
            continue
        if lines[j].speaker and lines[j].dia_speaker == tl.dia_speaker:
            return lines[j].speaker
        if lines[j].speaker:
            break  # different cluster — stop
    for j in range(idx + 1, min(idx + window, len(lines))):
        if lines[j].is_scene or not lines[j].text:
            continue
        if lines[j].speaker and lines[j].dia_speaker == tl.dia_speaker:
            return lines[j].speaker
        if lines[j].speaker:
            break
    return ""


def _infer_speaker(
    lines: list[TLine], idx: int,
    cmap: dict[str, ClusterMap],
    merged: dict[str, tuple[int, int, float, set[str]]],
) -> str:
    """Infer speaker for an unlabeled line using multiple signals."""
    tl = lines[idx]

    # Cluster signal
    cluster_char = ""
    m_votes, m_conf = 0, 0.0
    if tl.dia_speaker and tl.dia_speaker in cmap:
        cluster_char = cmap[tl.dia_speaker].character
        if cluster_char in merged:
            m_votes, _, m_conf, _ = merged[cluster_char]

    # 1. Strong merged cluster alone
    if cluster_char and m_conf >= 0.5 and m_votes >= MIN_CLUSTER_VOTES:
        return cluster_char

    # Context signals
    ctx = _context_speaker(lines, idx)
    nbr = _same_cluster_neighbor(lines, idx)

    # 2. Cluster + same-cluster neighbor agree
    if cluster_char and nbr == cluster_char:
        return cluster_char

    # 3. Cluster + context agree
    if cluster_char and ctx == cluster_char:
        return cluster_char

    # 4. Same-cluster neighbor alone
    if nbr:
        return nbr

    # 5. Context alone (both labeled neighbors agree)
    if ctx:
        return ctx

    return ""


def validate_and_fix(lines: list[TLine], cmap: dict[str, ClusterMap]) -> EpResult:
    r = EpResult()
    merged = build_merged_character_map(cmap)

    for i, tl in enumerate(lines):
        if tl.is_scene or not tl.text:
            continue
        r.total += 1

        if tl.speaker:
            r.labeled += 1
            if tl.dia_speaker and tl.dia_speaker in cmap:
                cm = cmap[tl.dia_speaker]
                # Check merged: line's speaker might match a different cluster
                if cm.character == tl.speaker:
                    tl.validation = "agree"
                    r.agree += 1
                elif tl.speaker in merged and tl.dia_speaker in merged[tl.speaker][3]:
                    # Speaker is in the set of clusters for this character
                    tl.validation = "agree"
                    r.agree += 1
                else:
                    tl.validation = "disagree"
                    r.disagree += 1
                    r.disagree_details.append({
                        "line": tl.line_num,
                        "text": tl.text[:60],
                        "transcript": tl.speaker,
                        "diarization": cm.character,
                        "cluster": tl.dia_speaker,
                        "conf": round(cm.confidence, 2),
                    })
            else:
                tl.validation = "unmatched"
        else:
            r.unlabeled += 1
            inferred = _infer_speaker(lines, i, cmap, merged)
            if inferred:
                tl.inferred = inferred
                r.fixed += 1
            else:
                r.unknown += 1

    r.n_clusters = len(cmap)
    r.cluster_info = {
        k: {"character": v.character, "votes": v.votes, "conf": round(v.confidence, 2)}
        for k, v in cmap.items()
    }
    return r


def format_fixed(lines: list[TLine], original: str) -> str:
    fixes = {tl.line_num: tl for tl in lines if tl.inferred}
    out = []
    for i, raw in enumerate(original.split("\n"), 1):
        if i in fixes:
            tl = fixes[i]
            out.append(f"{tl.inferred}:  {tl.text}")
        else:
            out.append(raw.rstrip())
    text = "\n".join(out)
    return text if text.endswith("\n") else text + "\n"


# ---------------------------------------------------------------------------
# Validate: episode discovery + processing
# ---------------------------------------------------------------------------


def discover_for_validate(root: Path, dia_dir: Path) -> list[dict]:
    """Find episodes with whisperX output, matched to transcripts."""
    eps = []
    for jp in sorted(dia_dir.glob("*.json")):
        if jp.name == PROGRESS_FILE:
            continue

        try:
            data = json.loads(jp.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue

        # Skip error-only files
        if "error" in data and "segments" not in data:
            continue

        eid = data.get("episode_id", jp.stem)

        # Find transcript
        tp = data.get("transcript_path", "")
        transcript = Path(tp) if tp and Path(tp).exists() else None

        if not transcript:
            m = EPISODE_RE.search(eid.replace(".", ""))
            if m:
                s, e = int(m.group(1)), int(m.group(2))
                code = eid.split(".")[0] if "." in eid else "AT"
                sdir = TRANSCRIPT_SERIES_DIRS.get(code, "Adventure Time")
                season_dir = root / sdir / f"Season {s:02d}"
                if season_dir.exists():
                    for txt in season_dir.glob(f"*S{s:02d}E{e:02d}*"):
                        transcript = txt
                        break

        if not transcript:
            continue

        m = EPISODE_RE.search(eid)
        season = int(m.group(1)) if m else 0

        eps.append({
            "episode_id": eid,
            "title": data.get("title", ""),
            "season": season,
            "dia_json": jp,
            "transcript": transcript,
            "segments": data.get("segments", []),
        })

    return eps


def validate_episode(
    ep: dict, label_dir: Path | None = None,
) -> tuple[EpResult, list[TLine], str]:
    """Validate a single episode using whisperX segments.

    If a label file exists (from embed-label), use it as the cluster->character
    mapping instead of deriving one from transcript majority voting.
    """
    eid = ep["episode_id"]
    original = ep["transcript"].read_text(encoding="utf-8")
    lines = parse_transcript(original)

    segments = ep["segments"]
    if not segments:
        r = EpResult(episode_id=eid, title=ep["title"], error="no segments")
        return r, lines, original

    # Match transcript lines to whisperX segments (text + speaker directly)
    match_to_segments(lines, segments)

    # Check for label file (independent cluster->character mapping)
    cmap = None
    if label_dir:
        label_path = label_dir / f"{eid}.json"
        if label_path.exists():
            label_data = json.loads(label_path.read_text(encoding="utf-8"))
            speaker_map = label_data.get("speaker_map", {})
            if speaker_map:
                # Build ClusterMap from the label file
                cmap = {}
                for cluster, character in speaker_map.items():
                    total = sum(
                        1 for tl in lines if tl.dia_speaker == cluster)
                    cmap[cluster] = ClusterMap(
                        cluster=cluster,
                        character=character,
                        votes=total,
                        total=total,
                    )

    # Fall back to transcript-derived voting
    if cmap is None:
        cmap = build_cluster_map(lines)

    result = validate_and_fix(lines, cmap)
    result.episode_id = eid
    result.title = ep["title"]
    result.matched = sum(1 for tl in lines if tl.w_start >= 0 and not tl.is_scene)

    return result, lines, original


# ---------------------------------------------------------------------------
# Embed: model management + audio + profiles
# ---------------------------------------------------------------------------

_ecapa_model = None


def _get_ecapa():
    """Lazy-load SpeechBrain ECAPA-TDNN model."""
    global _ecapa_model  # noqa: PLW0603
    if _ecapa_model is None:
        logging.getLogger("speechbrain").setLevel(logging.WARNING)
        from speechbrain.inference.speaker import EncoderClassifier
        from speechbrain.utils.fetching import LocalStrategy
        kwargs = {
            "source": ECAPA_SOURCE,
            "savedir": ECAPA_SAVEDIR,
            "run_opts": {"device": "cpu"},
        }
        # Windows: symlinks require admin, use COPY instead
        if sys.platform == "win32":
            kwargs["local_strategy"] = LocalStrategy.COPY
        _ecapa_model = EncoderClassifier.from_hparams(**kwargs)
    return _ecapa_model


def _extract_wav(video_path: Path, output_path: Path) -> bool:
    """Extract 16kHz mono WAV from video. Returns True on success."""
    subprocess.run(
        [find_tool("ffmpeg"), "-y", "-i", str(video_path), "-vn",
         "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
         str(output_path)],
        capture_output=True,
    )
    return output_path.exists() and output_path.stat().st_size > 100


def _load_wav(wav_path: Path):
    """Load WAV file. Returns (waveform, sample_rate)."""
    import torchaudio
    return torchaudio.load(str(wav_path))


def _slice_segment(waveform, sr: int, start: float, end: float):
    """Slice an audio segment from the full waveform."""
    s = int(start * sr)
    e = min(int(end * sr), waveform.shape[1])
    return waveform[:, s:e]


def _compute_embeddings_batch(segments: list):
    """Compute ECAPA-TDNN embeddings for audio tensors. Returns (N, 192) ndarray."""
    import torch

    model = _get_ecapa()
    max_len = max(s.shape[1] for s in segments)
    batch = torch.zeros(len(segments), max_len)
    wav_lens = torch.zeros(len(segments))
    for i, seg in enumerate(segments):
        batch[i, :seg.shape[1]] = seg[0]
        wav_lens[i] = seg.shape[1] / max_len

    with torch.no_grad():
        embeddings = model.encode_batch(batch, wav_lens)

    return embeddings.squeeze(1).numpy()


_alias_cache: dict[str, str] | None = None


def _load_alias_map() -> dict[str, str]:
    """Build speaker alias → canonical name map from characters.json."""
    global _alias_cache
    if _alias_cache is not None:
        return _alias_cache

    alias_map = dict(MANUAL_ALIASES)

    if ENCHIRIDION_CHARACTERS.exists():
        chars = json.loads(ENCHIRIDION_CHARACTERS.read_text(encoding="utf-8"))
        for char in chars:
            char_id = char["id"]
            char_name = char["name"]
            aliases = char.get("aliases", [])

            canonical = CANONICAL_SPEAKER.get(char_id, char_name)

            for variant in [char_name] + aliases:
                if variant in VOICE_SEPARATE:
                    continue
                if variant == canonical:
                    continue
                alias_map[variant] = canonical
    else:
        logger.warning("characters.json not found at %s", ENCHIRIDION_CHARACTERS)

    _alias_cache = alias_map
    return alias_map


def _resolve_speaker(name: str) -> str:
    """Resolve a transcript speaker name to its canonical form."""
    alias_map = _load_alias_map()
    return alias_map.get(name, name)


def _get_profile_name(character: str, season: int) -> str:
    """Return bucketed profile name (e.g. 'Finn_S01-S03') or plain name."""
    if character in SEASON_BUCKETS:
        for s, e in SEASON_BUCKETS[character]:
            if s <= season <= e:
                return f"{character}_S{s:02d}-S{e:02d}"
    return character


def _profile_dir(base: Path) -> Path:
    return base / "voice_profiles"


def _load_profile(path: Path) -> dict:
    """Load a voice profile from .npz file."""
    import numpy as np
    data = np.load(str(path), allow_pickle=False)
    meta_str = str(data["metadata"]) if "metadata" in data else "[]"
    return {
        "centroid": data["centroid"],
        "embeddings": data["embeddings"],
        "metadata": json.loads(meta_str),
    }


def _save_profile(
    path: Path, centroid, embeddings, metadata: list[dict],
) -> None:
    """Save a voice profile to .npz file."""
    import numpy as np
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(path),
        centroid=centroid,
        embeddings=embeddings,
        metadata=json.dumps(metadata),
    )


def _load_all_profiles(profile_dir: Path) -> dict:
    """Load all centroids from voice_profiles directory."""
    import numpy as np
    profiles = {}
    if not profile_dir.exists():
        return profiles
    for npz_path in sorted(profile_dir.glob("*.npz")):
        data = np.load(str(npz_path), allow_pickle=False)
        profiles[npz_path.stem] = data["centroid"]
    return profiles


def _filter_outliers(embeddings, threshold_std: float = 2.0):
    """Return boolean mask of embeddings to keep (remove outliers)."""
    import numpy as np
    if len(embeddings) < 4:
        return np.ones(len(embeddings), dtype=bool)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normed = embeddings / np.maximum(norms, 1e-10)
    sim_matrix = normed @ normed.T
    mean_sim = sim_matrix.mean(axis=1)
    threshold = mean_sim.mean() - threshold_std * mean_sim.std()
    return mean_sim >= threshold


def _compute_centroid(embeddings):
    """Compute L2-normalized centroid from embeddings with outlier filtering."""
    import numpy as np
    keep = _filter_outliers(embeddings)
    filtered = embeddings[keep]
    if len(filtered) == 0:
        filtered = embeddings
    centroid = filtered.mean(axis=0)
    norm = np.linalg.norm(centroid)
    if norm > 0:
        centroid /= norm
    return centroid


def _classify_segments(
    embeddings, profiles: dict, threshold: float = 0.5,
) -> list[dict]:
    """Classify embeddings against voice profiles by cosine similarity."""
    import numpy as np
    if not profiles:
        return [{"speaker": "", "confidence": 0.0, "scores": {}}] * len(embeddings)

    names = list(profiles.keys())
    centroids = np.stack([profiles[n] for n in names])

    e_norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    c_norms = np.linalg.norm(centroids, axis=1, keepdims=True)
    e_normed = embeddings / np.maximum(e_norms, 1e-10)
    c_normed = centroids / np.maximum(c_norms, 1e-10)

    sims = e_normed @ c_normed.T  # (N, C)

    results = []
    for i in range(len(embeddings)):
        best_idx = int(sims[i].argmax())
        best_sim = float(sims[i, best_idx])
        scores = {n: float(sims[i, j]) for j, n in enumerate(names)}
        scores = dict(sorted(scores.items(), key=lambda x: -x[1]))
        results.append({
            "speaker": names[best_idx] if best_sim >= threshold else "",
            "confidence": best_sim,
            "scores": scores,
        })
    return results


def _save_index(profile_dir: Path, profiles_meta: dict) -> None:
    """Save profile index JSON."""
    index = {
        "model": ECAPA_SOURCE,
        "embedding_dim": EMBED_DIM,
        "updated": datetime.now().isoformat(timespec="seconds"),
        "profiles": profiles_meta,
    }
    (profile_dir / "_index.json").write_text(
        json.dumps(index, indent=2, ensure_ascii=False), encoding="utf-8",
    )


def _select_season_profiles(profiles: dict, season: int) -> dict:
    """Select profiles appropriate for a given season.

    For season-bucketed profiles (e.g. Finn_S07-S10), include only the
    matching bucket.  Non-bucketed profiles always included.
    """
    active = {}
    # Track which base characters have a matching bucket
    matched_chars: set[str] = set()

    for pname, centroid in profiles.items():
        m = BUCKET_RE.match(pname)
        if m:
            char = m.group(1)
            s_start, s_end = int(m.group(2)), int(m.group(3))
            if s_start <= season <= s_end:
                active[pname] = centroid
                matched_chars.add(char)
        else:
            active[pname] = centroid

    # Include all buckets for chars with no matching bucket (fallback)
    for pname, centroid in profiles.items():
        m = BUCKET_RE.match(pname)
        if m and m.group(1) not in matched_chars:
            active[pname] = centroid

    return active


# ---------------------------------------------------------------------------
# CLI: process subcommand
# ---------------------------------------------------------------------------


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
        setup_logging(output_dir)

    project_root = Path(__file__).resolve().parent.parent
    all_episodes = discover_transcripts(project_root)
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

    logger.info("Processing %d episodes with %d workers (device=%s)",
                len(matched), args.workers, device)
    if device == "cuda" and args.workers > 1:
        logger.warning("Multiple GPU workers compete for VRAM; consider --workers 1")
    run_batch(matched, output_dir, args.workers, hf_token, args.whisper_model, device)


# ---------------------------------------------------------------------------
# CLI: validate subcommand
# ---------------------------------------------------------------------------


def cmd_validate(args: argparse.Namespace) -> None:
    # Force UTF-8 output on Windows
    if sys.stdout.encoding != "utf-8":
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    if sys.stderr.encoding != "utf-8":
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

    root = Path(__file__).resolve().parent.parent
    dia_dir = root / args.diarization_dir

    if not dia_dir.exists():
        print(f"Error: {dia_dir} not found", file=sys.stderr)
        sys.exit(1)

    episodes = discover_for_validate(root, dia_dir)
    print(f"Found {len(episodes)} episodes with diarization output")

    # Apply filters
    if args.episode:
        episodes = [e for e in episodes if e["episode_id"] in args.episode]
    if args.series:
        prefix = args.series.upper() + "."
        episodes = [e for e in episodes if e["episode_id"].startswith(prefix)]
    if args.season is not None:
        episodes = [e for e in episodes if e["season"] == args.season]
    if args.unlabeled_only:
        filtered = []
        for ep in episodes:
            text = ep["transcript"].read_text(encoding="utf-8")
            if "???:" in text or "[?]:" in text:
                filtered.append(ep)
        episodes = filtered

    # Resume: skip already-processed
    progress_path = dia_dir / PROGRESS_FILE
    progress = load_progress(progress_path)
    if not args.force:
        before = len(episodes)
        episodes = [e for e in episodes if e["episode_id"] not in progress["processed"]]
        skipped = before - len(episodes)
        if skipped:
            print(f"Skipping {skipped} already-processed (use --force to redo)")

    if not episodes:
        print("Nothing to process.")
        return

    if args.limit:
        episodes = episodes[:args.limit]

    # Check for label directory (from embed-label)
    label_dir = dia_dir / "labels"
    if not label_dir.exists():
        label_dir = None

    print(f"Processing {len(episodes)} episodes\n")

    total_agree = total_disagree = total_fixed = total_unknown = 0
    errors = []

    for i, ep in enumerate(episodes, 1):
        eid = ep["episode_id"]
        print(f"[{i:03d}/{len(episodes)}] {eid} {ep['title']}")

        try:
            result, lines, original = validate_episode(ep, label_dir)

            if result.error:
                print(f"  SKIP: {result.error}")
                errors.append(eid)
                progress["processed"][eid] = {
                    "ts": datetime.now().isoformat(), "error": result.error,
                }
                save_progress(progress_path, progress)
                continue

            m_pct = f"{result.matched}/{result.total}" if result.total else "0/0"
            print(f"  {result.total} dialogue ({result.labeled} labeled, "
                  f"{result.unlabeled} unlabeled) | matched {m_pct}")

            if result.cluster_info:
                parts = []
                for cid, info in sorted(result.cluster_info.items()):
                    parts.append(
                        f"{cid}={info['character']}({info['votes']},{info['conf']:.0%})"
                    )
                print(f"  Clusters: {', '.join(parts)}")
                # Show merged character view (combine split clusters)
                char_agg: dict[str, list] = defaultdict(list)
                for cid, info in result.cluster_info.items():
                    char_agg[info["character"]].append(
                        (cid, info["votes"], info["conf"]))
                multi = {c: v for c, v in char_agg.items() if len(v) > 1}
                if multi:
                    mparts = []
                    for char, clusters in sorted(multi.items()):
                        total_v = sum(v for _, v, _ in clusters)
                        cids = "+".join(c for c, _, _ in clusters)
                        mparts.append(f"{char}({total_v} votes via {cids})")
                    print(f"  Merged: {', '.join(mparts)}")

            print(f"  Validation: {result.agree} agree, {result.disagree} disagree"
                  + (f" | Fixed: {result.fixed}, unknown: {result.unknown}"
                     if result.unlabeled else ""))

            for d in result.disagree_details[:5]:
                print(f"    DISAGREE L{d['line']}: "
                      f"{d['transcript']} -> {d['diarization']} "
                      f"(cluster {d['cluster']}, conf {d['conf']:.0%})")
            if len(result.disagree_details) > 5:
                print(f"    ... and {len(result.disagree_details) - 5} more")

            fix_lines = [tl for tl in lines if tl.inferred]
            if args.write and fix_lines:
                new_text = format_fixed(lines, original)
                ep["transcript"].write_text(new_text, encoding="utf-8")
                print(f"  WROTE {len(fix_lines)} fixes to {ep['transcript'].name}")
            elif args.dry_run and fix_lines:
                for tl in fix_lines[:10]:
                    print(f"    FIX L{tl.line_num}: -> {tl.inferred}:  {tl.text[:50]}")
                if len(fix_lines) > 10:
                    print(f"    ... and {len(fix_lines) - 10} more fixes")

            total_agree += result.agree
            total_disagree += result.disagree
            total_fixed += result.fixed
            total_unknown += result.unknown

            progress["processed"][eid] = {
                "ts": datetime.now().isoformat(),
                "total": result.total,
                "labeled": result.labeled,
                "unlabeled": result.unlabeled,
                "matched": result.matched,
                "agree": result.agree,
                "disagree": result.disagree,
                "fixed": result.fixed,
                "unknown": result.unknown,
                "clusters": result.cluster_info,
                "disagreements": result.disagree_details,
            }
            save_progress(progress_path, progress)

        except Exception as e:
            print(f"  ERROR: {e}", file=sys.stderr)
            errors.append(eid)
            progress["processed"][eid] = {
                "ts": datetime.now().isoformat(), "error": str(e),
            }
            save_progress(progress_path, progress)

    total_processed = len(episodes) - len(errors)
    print(f"\n{'=' * 60}")
    print(f"Processed: {total_processed}/{len(episodes)}"
          + (f" ({len(errors)} errors)" if errors else ""))
    print(f"Validation: {total_agree} agree, {total_disagree} disagree")
    if total_fixed or total_unknown:
        print(f"Fixes: {total_fixed} applied, {total_unknown} still unknown")
    if errors:
        print(f"Failed: {', '.join(errors)}")


# ---------------------------------------------------------------------------
# CLI: embed-clusters subcommand
# ---------------------------------------------------------------------------


def _embed_clusters_one(
    ep_id: str, data: dict, video_path: Path, dia_dir: Path, args: argparse.Namespace,
) -> str:
    """Build per-cluster embeddings for a single episode. Returns report text."""
    import numpy as np

    segments = data.get("segments", [])
    by_cluster: dict[str, list[dict]] = defaultdict(list)
    for seg in segments:
        spk = seg.get("speaker", "")
        if spk:
            by_cluster[spk].append(seg)

    lines: list[str] = []
    lines.append(f"Episode: {ep_id} ({data.get('title', '')})")
    lines.append(f"{len(by_cluster)} speaker clusters, {len(segments)} total segments\n")

    tmp_dir = tempfile.mkdtemp(prefix="embed_clusters_")
    wav_path = Path(tmp_dir) / f"{ep_id}.wav"

    try:
        if not _extract_wav(video_path, wav_path):
            lines.append("ERROR: WAV extraction failed")
            return "\n".join(lines)

        waveform, sr = _load_wav(wav_path)

        cluster_centroids = {}
        cluster_embeddings = {}
        cluster_meta = {}

        for spk_id in sorted(by_cluster.keys()):
            segs = by_cluster[spk_id]

            eligible = []
            for seg in segs:
                dur = seg["end"] - seg["start"]
                if args.min_duration <= dur <= args.max_duration:
                    eligible.append(seg)

            total_time = sum(s["end"] - s["start"] for s in segs)

            audio_segments = []
            valid_segs = []
            for seg in eligible:
                sliced = _slice_segment(waveform, sr, seg["start"], seg["end"])
                if sliced.shape[1] >= sr * 0.5:
                    audio_segments.append(sliced)
                    valid_segs.append(seg)

            if audio_segments:
                embs = _compute_embeddings_batch(audio_segments)
                centroid = _compute_centroid(embs)
                cluster_centroids[spk_id] = centroid
                cluster_embeddings[spk_id] = embs
            else:
                embs = np.zeros((0, EMBED_DIM), dtype=np.float32)
                cluster_embeddings[spk_id] = embs

            samples = []
            for seg in segs[:args.samples]:
                text = seg.get("text", "").strip()
                if text:
                    samples.append(text)

            cluster_meta[spk_id] = {
                "total_segments": len(segs),
                "embedded_segments": len(valid_segs),
                "total_time": round(total_time, 1),
                "samples": samples,
            }

            lines.append(f"\n{spk_id} ({len(segs)} segs, {total_time:.1f}s, "
                         f"{len(valid_segs)} embedded)")
            for s in samples:
                lines.append(f'  "{s[:80]}"')

        # Pairwise cluster similarity
        if len(cluster_centroids) >= 2:
            names = sorted(cluster_centroids.keys())
            centroids_arr = np.stack([cluster_centroids[n] for n in names])
            c_norms = np.linalg.norm(centroids_arr, axis=1, keepdims=True)
            c_normed = centroids_arr / np.maximum(c_norms, 1e-10)
            sim_matrix = c_normed @ c_normed.T

            lines.append(f"\n{'=' * 60}")
            lines.append("PAIRWISE SIMILARITY:")
            pairs = []
            for i in range(len(names)):
                for j in range(i + 1, len(names)):
                    sim = float(sim_matrix[i, j])
                    pairs.append((names[i], names[j], sim))
            pairs.sort(key=lambda x: -x[2])

            has_likely_split = False
            for a, b, sim in pairs:
                if sim >= 0.75:
                    has_likely_split = True
                    lines.append(f"  {a} <-> {b}: {sim:.3f}  <- likely same character")
                elif sim >= 0.60:
                    lines.append(f"  {a} <-> {b}: {sim:.3f}  <- possible match")

            if not has_likely_split:
                for a, b, sim in pairs[:3]:
                    lines.append(f"  {a} <-> {b}: {sim:.3f}")

        # Save cluster embeddings
        cluster_dir = dia_dir / "clusters"
        cluster_dir.mkdir(parents=True, exist_ok=True)
        save_path = cluster_dir / f"{ep_id}.npz"

        arrays = {}
        for spk_id in sorted(cluster_embeddings.keys()):
            arrays[f"{spk_id}_embeddings"] = cluster_embeddings[spk_id]
            if spk_id in cluster_centroids:
                arrays[f"{spk_id}_centroid"] = cluster_centroids[spk_id]

        arrays["_meta"] = json.dumps({
            "episode_id": ep_id,
            "title": data.get("title", ""),
            "season": data.get("season", 0),
            "clusters": cluster_meta,
        })

        np.savez_compressed(str(save_path), **arrays)
        lines.append(f"\nSaved cluster embeddings to {save_path}")

    finally:
        if wav_path.exists():
            wav_path.unlink(missing_ok=True)
        try:
            Path(tmp_dir).rmdir()
        except OSError:
            pass

    return "\n".join(lines)


def cmd_embed_clusters(args: argparse.Namespace) -> None:
    """Build per-cluster embeddings from anonymous speakers and print report."""
    if sys.stdout.encoding != "utf-8":
        sys.stdout = io.TextIOWrapper(
            sys.stdout.buffer, encoding="utf-8", errors="replace")

    root = Path(__file__).resolve().parent.parent
    dia_dir = root / args.diarization_dir
    force = getattr(args, "force", False)
    limit = getattr(args, "limit", None)

    # Discover eligible episodes
    if args.episode:
        # Specific episodes requested
        ep_ids = args.episode if isinstance(args.episode, list) else [args.episode]
    else:
        # Auto-discover: have JSON, missing NPZ (unless --force)
        ep_ids = []
        cluster_dir = dia_dir / "clusters"
        existing_npz = (
            {p.stem for p in cluster_dir.glob("*.npz")} if cluster_dir.is_dir() else set()
        )
        for jp in sorted(dia_dir.glob("*.json")):
            if jp.name == PROGRESS_FILE:
                continue
            eid = jp.stem
            if force or eid not in existing_npz:
                ep_ids.append(eid)

    # Apply series/season filters
    series_filter = getattr(args, "series", "all")
    season_filter = getattr(args, "season", None)
    if series_filter and series_filter != "all":
        code = series_filter.upper()
        ep_ids = [e for e in ep_ids if e.startswith(code + ".")]
    if season_filter is not None:
        pattern = f"S{season_filter:02d}E"
        ep_ids = [e for e in ep_ids if pattern in e]

    if limit:
        ep_ids = ep_ids[:limit]

    if not ep_ids:
        print("No episodes to process.")
        return

    # Resolve video paths
    video_dirs = getattr(args, "video_dirs", DEFAULT_VIDEO_DIRS)
    video_index = scan_videos(video_dirs)
    all_transcripts = discover_transcripts(root)
    match_episodes(all_transcripts, video_index)
    transcript_map = {ep.episode_id: ep for ep in all_transcripts}

    # Load ECAPA model once before the loop
    print("Loading ECAPA-TDNN model...")
    _get_ecapa()

    # Create report directory
    report_dir = dia_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    total = len(ep_ids)
    succeeded = 0
    t0 = time.monotonic()

    for idx, ep_id in enumerate(ep_ids, 1):
        json_path = dia_dir / f"{ep_id}.json"
        if not json_path.exists():
            print(f"[{idx:03d}/{total}] {ep_id} -- SKIP (no JSON)", file=sys.stderr)
            continue

        data = json.loads(json_path.read_text(encoding="utf-8"))
        segments = data.get("segments", [])
        if not segments:
            print(f"[{idx:03d}/{total}] {ep_id} -- SKIP (no segments)", file=sys.stderr)
            continue

        # Resolve video path: try JSON, then transcript match, then video index
        video_path = data.get("video_path", "")
        if not video_path or not Path(video_path).exists():
            ep = transcript_map.get(ep_id)
            if ep and ep.video_path:
                video_path = str(ep.video_path)

        if not video_path or not Path(video_path).exists():
            print(f"[{idx:03d}/{total}] {ep_id} -- SKIP (no video)", file=sys.stderr)
            continue

        elapsed = time.monotonic() - t0
        title = data.get("title", "")
        print(f"\n[{idx:03d}/{total}] {ep_id} {title}")

        report = _embed_clusters_one(ep_id, data, Path(video_path), dia_dir, args)
        print(report)

        # Save report file
        report_path = report_dir / f"{ep_id}.txt"
        report_path.write_text(report, encoding="utf-8")
        succeeded += 1

    elapsed = time.monotonic() - t0
    print(f"\n{'=' * 60}")
    print(f"Embed-clusters complete: {succeeded}/{total} in {format_duration(elapsed)}")


# ---------------------------------------------------------------------------
# CLI: embed-label subcommand
# ---------------------------------------------------------------------------


def cmd_embed_label(args: argparse.Namespace) -> None:
    """Save cluster->character mapping and merge into character profiles."""
    import numpy as np

    if sys.stdout.encoding != "utf-8":
        sys.stdout = io.TextIOWrapper(
            sys.stdout.buffer, encoding="utf-8", errors="replace")

    root = Path(__file__).resolve().parent.parent
    dia_dir = root / args.diarization_dir
    pdir = _profile_dir(dia_dir)

    ep_id = args.episode

    # Parse --map args: SPEAKER_00=Finn SPEAKER_01=Jake ...
    speaker_map: dict[str, str] = {}
    skips: set[str] = set()
    if args.skip:
        skips = set(args.skip)

    for mapping in args.map:
        if "=" not in mapping:
            print(f"Error: invalid mapping '{mapping}' — "
                  f"use CLUSTER=Character format", file=sys.stderr)
            sys.exit(1)
        cluster, character = mapping.split("=", 1)
        canonical = _resolve_speaker(character)
        speaker_map[cluster] = canonical

    if not speaker_map:
        print("Error: no mappings provided", file=sys.stderr)
        sys.exit(1)

    # Load cluster embeddings
    cluster_path = dia_dir / "clusters" / f"{ep_id}.npz"
    if not cluster_path.exists():
        print(f"Error: {cluster_path} not found. "
              f"Run 'embed-clusters --episode {ep_id}' first.",
              file=sys.stderr)
        sys.exit(1)

    cluster_data = np.load(str(cluster_path), allow_pickle=False)
    meta_str = str(cluster_data.get("_meta", "{}"))
    meta = json.loads(meta_str)
    season = meta.get("season", 0)

    print(f"Episode: {ep_id} ({meta.get('title', '')})")
    print(f"Mapping {len(speaker_map)} clusters:")
    for cluster, character in sorted(speaker_map.items()):
        print(f"  {cluster} -> {character}")
    if skips:
        print(f"Skipping: {', '.join(sorted(skips))}")

    # Merge cluster embeddings into character profiles
    for cluster, character in sorted(speaker_map.items()):
        if cluster in skips:
            continue

        emb_key = f"{cluster}_embeddings"
        if emb_key not in cluster_data:
            print(f"  Warning: no embeddings for {cluster}")
            continue

        new_embeddings = cluster_data[emb_key]
        if len(new_embeddings) == 0:
            print(f"  {cluster} -> {character}: no embeddings to merge")
            continue

        profile_name = _get_profile_name(character, season)
        profile_path = pdir / f"{profile_name}.npz"

        # Build metadata for new samples
        new_metadata = [{
            "episode": ep_id,
            "season": season,
            "cluster": cluster,
            "sample_idx": i,
        } for i in range(len(new_embeddings))]

        if profile_path.exists():
            existing = _load_profile(profile_path)
            all_embeddings = np.vstack(
                [existing["embeddings"], new_embeddings])
            all_metadata = existing["metadata"] + new_metadata
            print(f"  {cluster} -> {profile_name}: "
                  f"+{len(new_embeddings)} samples "
                  f"(total {len(all_embeddings)})")
        else:
            all_embeddings = new_embeddings
            all_metadata = new_metadata
            print(f"  {cluster} -> {profile_name}: "
                  f"{len(new_embeddings)} samples (new)")

        centroid = _compute_centroid(all_embeddings)
        pdir.mkdir(parents=True, exist_ok=True)
        _save_profile(profile_path, centroid, all_embeddings, all_metadata)

    # Save label file
    labels_dir = dia_dir / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    label_path = labels_dir / f"{ep_id}.json"
    label_data = {
        "episode_id": ep_id,
        "title": meta.get("title", ""),
        "season": season,
        "speaker_map": speaker_map,
        "skipped": sorted(skips),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    label_path.write_text(
        json.dumps(label_data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\nSaved label mapping to {label_path}")

    # Update profile index
    profiles_meta = {}
    for npz_path in pdir.glob("*.npz"):
        prof = _load_profile(npz_path)
        episodes_in = list({
            m.get("episode", "") for m in prof["metadata"]
        })
        profiles_meta[npz_path.stem] = {
            "samples": len(prof["embeddings"]),
            "episodes": sorted(e for e in episodes_in if e),
            "updated": datetime.now().isoformat(timespec="seconds"),
        }
    _save_index(pdir, profiles_meta)
    print(f"Updated index: {len(profiles_meta)} profiles")


# ---------------------------------------------------------------------------
# Progress tracking
# ---------------------------------------------------------------------------


def load_progress(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"processed": {}, "started": datetime.now().isoformat()}


def save_progress(path: Path, data: dict) -> None:
    data["updated"] = datetime.now().isoformat()
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------


def setup_logging(output_dir: Path) -> None:
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(output_dir / "whisperx_diarize.log", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(sh)


# ---------------------------------------------------------------------------
# CLI: status subcommand
# ---------------------------------------------------------------------------


def discover_pipeline_status(
    root: Path, dia_dir: Path, video_index: dict | None = None,
) -> list[dict]:
    """Check pipeline stage completion for every transcript."""
    transcripts = discover_transcripts(root)
    if video_index:
        match_episodes(transcripts, video_index)

    jsons = {p.stem for p in dia_dir.glob("*.json") if p.stem != PROGRESS_FILE}
    cluster_dir = dia_dir / "clusters"
    npzs = {p.stem for p in cluster_dir.glob("*.npz")} if cluster_dir.is_dir() else set()
    label_dir = dia_dir / "labels"
    labels = {p.stem for p in label_dir.glob("*.json")} if label_dir.is_dir() else set()

    results = []
    for ep in transcripts:
        eid = ep.episode_id
        results.append({
            "episode_id": eid,
            "series": ep.series,
            "season": ep.season,
            "title": ep.title,
            "has_video": ep.video_path is not None,
            "has_json": eid in jsons,
            "has_clusters": eid in npzs,
            "has_labels": eid in labels,
        })
    return results


def cmd_status(args: argparse.Namespace) -> None:
    """Show pipeline progress across all episodes."""
    if sys.stdout.encoding != "utf-8":
        sys.stdout = io.TextIOWrapper(
            sys.stdout.buffer, encoding="utf-8", errors="replace")

    root = Path(__file__).resolve().parent.parent
    dia_dir = root / args.diarization_dir

    # Try video scan (graceful skip if drives offline)
    video_index = None
    try:
        video_index = scan_videos(args.video_dirs)
    except Exception:
        pass

    all_eps = discover_pipeline_status(root, dia_dir, video_index)

    # Apply filters
    if args.series != "all":
        code = args.series.upper()
        all_eps = [e for e in all_eps if e["series"] == code]
    if args.season is not None:
        all_eps = [e for e in all_eps if e["season"] == args.season]

    if not all_eps:
        print("No episodes found matching filters.")
        return

    total = len(all_eps)
    n_video = sum(1 for e in all_eps if e["has_video"])
    n_json = sum(1 for e in all_eps if e["has_json"])
    n_clusters = sum(1 for e in all_eps if e["has_clusters"])
    n_labels = sum(1 for e in all_eps if e["has_labels"])
    n_complete = sum(1 for e in all_eps if e["has_labels"])

    # Header
    series_label = args.series.upper() if args.series != "all" else "all series"
    season_label = f" S{args.season:02d}" if args.season else ""
    print(f"\nPipeline Status ({series_label}{season_label}, {total} episodes)")
    print("=" * 60)
    print(f"{'Stage':<14} {'Done':>6}  {'Pending':>7}")
    print(f"{'video':.<14} {n_video:>6}  {total - n_video:>7}")
    print(f"{'process':.<14} {n_json:>6}  {n_video - n_json:>7}  (of {n_video} with video)")
    print(f"{'clusters':.<14} {n_clusters:>6}  {n_json - n_clusters:>7}  (of {n_json} processed)")
    print(f"{'labels':.<14} {n_labels:>6}  {n_clusters - n_labels:>7}  (of {n_clusters} clustered)")
    print(f"\nFully complete: {n_complete}/{total} ({100*n_complete/total:.1f}%)")

    # Filtered list based on --needs
    needs = getattr(args, "needs", None)
    if needs:
        if needs == "process":
            pending = [e for e in all_eps if e["has_video"] and not e["has_json"]]
            label = "needing processing"
        elif needs == "clusters":
            pending = [e for e in all_eps if e["has_json"] and not e["has_clusters"]]
            label = "needing clustering"
        elif needs == "labels":
            pending = [e for e in all_eps if e["has_clusters"] and not e["has_labels"]]
            label = "needing labels"
        else:  # "any"
            pending = [e for e in all_eps
                       if not e["has_labels"] and e["has_video"]]
            label = "incomplete (with video)"

        if pending:
            print(f"\nEpisodes {label} ({len(pending)}):")
            for e in pending[:50]:
                print(f"  {e['episode_id']}  {e['title']}")
            if len(pending) > 50:
                print(f"  ... and {len(pending) - 50} more")
        else:
            print(f"\nNo episodes {label}.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Combined whisperX diarization + validation for "
                    "Adventure Time transcripts.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # process subcommand
    p_proc = sub.add_parser("process", help="Run whisperX pipeline on video files")
    p_proc.add_argument(
        "--video-dirs", nargs="+", type=Path, default=DEFAULT_VIDEO_DIRS,
        help="Directories to search for video files",
    )
    p_proc.add_argument("--output-dir", default="diarization", help="Output directory")
    p_proc.add_argument("--workers", type=int, default=2, help="Parallel workers (default: 2)")
    p_proc.add_argument("--whisper-model", default=WHISPER_MODEL, help="Whisper model")
    p_proc.add_argument("--series", choices=["all", "at", "dl", "fc"], default="all")
    p_proc.add_argument("--season", type=int, help="Filter by season")
    p_proc.add_argument("--episode", nargs="+", help="Filter by episode ID(s) (e.g. AT.S09E11 AT.S08E08)")
    p_proc.add_argument("--device", choices=["cpu", "cuda", "auto"], default="auto",
                        help="Device for inference (default: auto-detect)")
    p_proc.add_argument("--dry-run", action="store_true", help="Preview mapping only")
    p_proc.add_argument("--force", action="store_true", help="Re-process existing output")

    # validate subcommand
    p_val = sub.add_parser("validate", help="Validate/fix transcript speaker labels")
    p_val.add_argument("--episode", nargs="+", help="Filter by episode ID(s) (e.g. AT.S09E11 AT.S08E08)")
    p_val.add_argument("--series", help="Filter by series: at, dl, fc")
    p_val.add_argument("--season", type=int, help="Filter by season")
    p_val.add_argument("--unlabeled-only", action="store_true", help="Only ??? episodes")
    p_val.add_argument("--write", action="store_true", help="Apply fixes to transcripts")
    p_val.add_argument("--dry-run", action="store_true", help="Preview fixes")
    p_val.add_argument("--force", action="store_true", help="Reprocess validated episodes")
    p_val.add_argument("--limit", type=int, help="Max episodes per run")
    p_val.add_argument("--diarization-dir", default="diarization", help="JSON directory")

    # embed-clusters subcommand
    p_eclust = sub.add_parser(
        "embed-clusters",
        help="Build per-cluster embeddings from anonymous speakers")
    p_eclust.add_argument(
        "--episode", nargs="+",
        help="Episode ID(s). Omit to auto-discover unprocessed episodes.")
    p_eclust.add_argument("--series", choices=["all", "at", "dl", "fc"], default="all")
    p_eclust.add_argument("--season", type=int, help="Filter by season")
    p_eclust.add_argument(
        "--video-dirs", nargs="+", type=Path, default=DEFAULT_VIDEO_DIRS,
        help="Directories to search for video files",
    )
    p_eclust.add_argument("--force", action="store_true", help="Re-process existing clusters")
    p_eclust.add_argument("--limit", type=int, help="Max episodes to process")
    p_eclust.add_argument(
        "--min-duration", type=float, default=MIN_EMBED_DURATION,
        help="Min segment duration in seconds (default: 1.5)")
    p_eclust.add_argument(
        "--max-duration", type=float, default=MAX_EMBED_DURATION,
        help="Max segment duration in seconds (default: 15)")
    p_eclust.add_argument(
        "--samples", type=int, default=5,
        help="Sample dialogue lines per cluster (default: 5)")
    p_eclust.add_argument(
        "--diarization-dir", default="diarization", help="JSON directory")

    # embed-label subcommand
    p_elabel = sub.add_parser(
        "embed-label",
        help="Save cluster->character mapping and merge into profiles")
    p_elabel.add_argument(
        "--episode", required=True, help="Episode ID (e.g. AT.S08E08)")
    p_elabel.add_argument(
        "--map", nargs="+", required=True,
        help="Cluster=Character mappings (e.g. SPEAKER_00=Finn)")
    p_elabel.add_argument(
        "--skip", nargs="*", default=[],
        help="Clusters to skip (noise, music, etc.)")
    p_elabel.add_argument(
        "--diarization-dir", default="diarization", help="JSON directory")

    # status subcommand
    p_status = sub.add_parser("status", help="Show pipeline progress across all episodes")
    p_status.add_argument("--series", choices=["all", "at", "dl", "fc"], default="all")
    p_status.add_argument("--season", type=int, help="Filter by season")
    p_status.add_argument(
        "--video-dirs", nargs="+", type=Path, default=DEFAULT_VIDEO_DIRS,
        help="Directories to search for video files",
    )
    p_status.add_argument("--diarization-dir", default="diarization", help="JSON directory")
    p_status.add_argument(
        "--needs", choices=["process", "clusters", "labels", "any"],
        help="Show only episodes needing this stage",
    )

    args = parser.parse_args()

    if args.command == "process":
        cmd_process(args)
    elif args.command == "validate":
        cmd_validate(args)
    elif args.command == "embed-clusters":
        cmd_embed_clusters(args)
    elif args.command == "embed-label":
        cmd_embed_label(args)
    elif args.command == "status":
        cmd_status(args)


if __name__ == "__main__":
    main()
