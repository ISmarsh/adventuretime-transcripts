#!/usr/bin/env python3
"""Extract speaker labels for unlabeled transcript lines using SDH subtitles and Vision AI.

Merges split subtitle lines, mines speaker labels from SDH subtitle tracks,
applies rule-based heuristics, and optionally uses Claude Vision API (batched
by scene) to identify speakers from video frames.

Requirements:
    ffmpeg and ffprobe must be on PATH
    pip install anthropic  (only needed with --vision)
    ANTHROPIC_API_KEY env var (only needed with --vision)

Usage:
    python extract_speakers.py <transcript> --video <path>              # SDH extraction
    python extract_speakers.py <transcript> --video <path> --scan-only  # list tracks
    python extract_speakers.py <transcript> --srt <path>                # pre-extracted SRT
    python extract_speakers.py <transcript> --video <path> --vision     # + Claude Vision
"""

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path

# Speaker label pattern: capitalized name(s) followed by colon at line start
SPEAKER_RE = re.compile(r"^([A-Z][A-Za-z \u2019'.\-()]+):\s")

# SDH speaker patterns: [SPEAKER], SPEAKER:, (SPEAKER)
SDH_SPEAKER_RE = re.compile(
    r"^\[([A-Z][A-Z .']+)\]\s*(.*)|"
    r"^([A-Z][A-Z .']+):\s*(.*)|"
    r"^\(([A-Z][A-Z .']+)\)\s*(.*)"
)

MUSIC_RE = re.compile(r"^\u266a|^\u266b")

MIN_MATCH_RATIO = 0.6

# Max lines per Vision API batch
VISION_BATCH_SIZE = 10

# Max frames to send per batch (deduplicated)
MAX_FRAMES_PER_BATCH = 4


@dataclass
class SrtEntry:
    index: int
    start_time: str
    end_time: str
    speaker: str
    text: str
    start_seconds: float = 0.0


@dataclass
class TranscriptLine:
    line_num: int
    speaker: str
    text: str
    original: str
    is_scene: bool = False
    matched_srt: SrtEntry | None = None
    vision_speaker: str = ""
    confidence: float = 0.0
    source: str = ""  # "sdh", "rule", "vision", or ""


@dataclass
class SubtitleTrack:
    index: int
    codec: str
    language: str
    title: str
    is_text: bool = field(default=False)


def find_tool(name: str) -> str:
    """Find an external tool on PATH."""
    path = shutil.which(name)
    if not path:
        print(f"Error: {name} not found on PATH", file=sys.stderr)
        sys.exit(1)
    return path


def srt_time_to_seconds(time_str: str) -> float:
    """Convert SRT timestamp (HH:MM:SS,mmm) to seconds."""
    time_str = time_str.replace(",", ".")
    parts = time_str.split(":")
    return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])


# ---------------------------------------------------------------------------
# Subtitle track scanning
# ---------------------------------------------------------------------------


def scan_subtitle_tracks(video_path: str) -> list[SubtitleTrack]:
    """Use ffprobe to list subtitle tracks in a video file."""
    ffprobe = find_tool("ffprobe")
    result = subprocess.run(
        [
            ffprobe, "-v", "error",
            "-show_entries", "stream=index,codec_name,codec_type:stream_tags=title,language",
            "-of", "json", video_path,
        ],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"ffprobe error: {result.stderr}", file=sys.stderr)
        return []

    data = json.loads(result.stdout)
    tracks = []
    text_codecs = {"srt", "subrip", "ass", "ssa", "mov_text", "webvtt"}

    for stream in data.get("streams", []):
        if stream.get("codec_type") != "subtitle":
            continue
        tags = stream.get("tags", {})
        codec = stream.get("codec_name", "unknown")
        tracks.append(SubtitleTrack(
            index=stream["index"],
            codec=codec,
            language=tags.get("language", "und"),
            title=tags.get("title", ""),
            is_text=codec in text_codecs,
        ))

    return tracks


def extract_subtitle_track(video_path: str, track_index: int, output_path: str) -> bool:
    """Extract a subtitle track to SRT using ffmpeg."""
    ffmpeg = find_tool("ffmpeg")
    result = subprocess.run(
        [ffmpeg, "-y", "-i", video_path, "-map", f"0:{track_index}", output_path],
        capture_output=True, text=True,
    )
    return result.returncode == 0


# ---------------------------------------------------------------------------
# SRT parsing
# ---------------------------------------------------------------------------


def parse_srt(srt_path: str) -> list[SrtEntry]:
    """Parse SRT file, extracting speaker labels from SDH formatting."""
    text = Path(srt_path).read_text(encoding="utf-8", errors="replace")
    entries = []
    blocks = re.split(r"\n\n+", text.strip())

    for block in blocks:
        lines = block.strip().split("\n")
        if len(lines) < 3:
            continue

        try:
            index = int(lines[0].strip())
        except ValueError:
            continue

        time_line = lines[1].strip()
        if " --> " not in time_line:
            continue
        start_time, end_time = time_line.split(" --> ")

        dialogue = " ".join(lines[2:]).strip()
        dialogue = re.sub(r"<[^>]+>", "", dialogue)
        # Strip SRT formatting tags like {\an8}
        dialogue = re.sub(r"\{\\[^}]+\}", "", dialogue)

        speaker = ""
        m = SDH_SPEAKER_RE.match(dialogue)
        if m:
            groups = m.groups()
            for i in range(0, len(groups), 2):
                if groups[i] is not None:
                    speaker = groups[i].strip().title()
                    dialogue = groups[i + 1].strip()
                    break

        entries.append(SrtEntry(
            index=index,
            start_time=start_time.strip(),
            end_time=end_time.strip(),
            speaker=speaker,
            text=dialogue,
            start_seconds=srt_time_to_seconds(start_time.strip()),
        ))

    return entries


# ---------------------------------------------------------------------------
# Transcript parsing and line merging
# ---------------------------------------------------------------------------


def merge_subtitle_lines(text: str) -> str:
    """Merge lines that were split across subtitle timing boundaries."""
    lines = text.split("\n")
    blocks: list[list[str]] = []
    current: list[str] = []

    for line in lines:
        stripped = line.strip()
        if stripped == "":
            if current:
                blocks.append(current)
                current = []
            blocks.append([""])
        else:
            current.append(stripped)

    if current:
        blocks.append(current)

    result: list[str] = []
    for block in blocks:
        if block == [""]:
            result.append("")
            continue

        if len(block) == 1:
            result.append(block[0])
            continue

        # Multi-speaker subtitle: all lines dash-prefixed
        if all(line.startswith("- ") or line.startswith("\u2014") for line in block):
            for line in block:
                cleaned = re.sub(r"^[-\u2014]\s*", "", line)
                if cleaned:
                    result.append(cleaned)
            continue

        # Don't merge across speaker label or bracket boundaries
        merged_parts: list[str] = []
        current_group: list[str] = []

        for line in block:
            is_label = bool(SPEAKER_RE.match(line))
            is_bracket = line.startswith("[")
            is_music = bool(MUSIC_RE.match(line))

            if (is_label or is_bracket or is_music) and current_group:
                merged_parts.append(" ".join(current_group))
                current_group = []

            current_group.append(line)

        if current_group:
            merged_parts.append(" ".join(current_group))

        result.extend(merged_parts)

    return "\n".join(result)


def parse_transcript(text: str) -> list[TranscriptLine]:
    """Parse transcript into structured lines."""
    lines = []
    for i, raw in enumerate(text.split("\n"), 1):
        stripped = raw.strip()
        if not stripped:
            continue

        is_scene = stripped.startswith("[")
        m = SPEAKER_RE.match(stripped)
        if m:
            speaker = m.group(1).strip()
            dialogue = stripped[m.end():].strip()
        else:
            speaker = ""
            dialogue = stripped

        lines.append(TranscriptLine(
            line_num=i,
            speaker=speaker,
            text=dialogue,
            original=stripped,
            is_scene=is_scene,
        ))

    return lines


# ---------------------------------------------------------------------------
# Fuzzy matching
# ---------------------------------------------------------------------------


def match_transcript_to_srt(
    transcript_lines: list[TranscriptLine],
    srt_entries: list[SrtEntry],
) -> list[TranscriptLine]:
    """Match transcript lines to SRT entries using fuzzy text matching."""
    srt_idx = 0

    for tline in transcript_lines:
        if tline.is_scene:
            continue

        text_to_match = tline.text if not tline.speaker else tline.text
        if not text_to_match:
            continue

        best_ratio = 0.0
        best_entry = None

        search_start = max(0, srt_idx - 5)
        search_end = min(len(srt_entries), srt_idx + 20)

        for j in range(search_start, search_end):
            entry = srt_entries[j]
            if not entry.text:
                continue
            ratio = SequenceMatcher(None, text_to_match.lower(), entry.text.lower()).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_entry = entry

        if best_entry and best_ratio >= MIN_MATCH_RATIO:
            tline.matched_srt = best_entry
            tline.confidence = best_ratio
            if best_entry.speaker and not tline.speaker:
                tline.speaker = best_entry.speaker
                tline.source = "sdh"

            srt_idx = max(srt_idx, srt_entries.index(best_entry) + 1)

    return transcript_lines


# ---------------------------------------------------------------------------
# Rule-based speaker inference
# ---------------------------------------------------------------------------


def infer_speakers_rules(transcript_lines: list[TranscriptLine]) -> list[TranscriptLine]:
    """Apply heuristic rules to infer speakers without external data."""
    # Build list of known speakers from labeled lines
    known_speakers: set[str] = set()
    for tline in transcript_lines:
        if tline.speaker:
            known_speakers.add(tline.speaker)

    # Self-reference: "I'm [Name]", "My name is [Name]", "It's me, [Name]"
    self_ref_re = re.compile(
        r"(?:I'm |I am |My name is |It's me,? |This is )([A-Z][a-z]+)",
        re.IGNORECASE,
    )

    for i, tline in enumerate(transcript_lines):
        if tline.speaker or tline.is_scene:
            continue

        text = tline.text

        # Self-reference heuristic
        m = self_ref_re.search(text)
        if m:
            name = m.group(1).title()
            if name in known_speakers:
                tline.speaker = name
                tline.source = "rule"
                tline.confidence = 0.6
                continue

        # Address-response heuristic: if prev line addresses a name, this line might be that name
        if i > 0:
            prev = transcript_lines[i - 1]
            if prev.speaker and not prev.is_scene:
                # Check if prev line ends with addressing someone
                addr_match = re.search(
                    r",\s*([A-Z][a-z]+)[.!?]*$", prev.text,
                )
                if addr_match:
                    addressed = addr_match.group(1).title()
                    if addressed in known_speakers and addressed != prev.speaker:
                        tline.speaker = addressed
                        tline.source = "rule"
                        tline.confidence = 0.5
                        continue

        # Turn-taking in two-character scenes: find surrounding labeled lines
        prev_speaker = ""
        next_speaker = ""
        for j in range(i - 1, max(i - 5, -1), -1):
            pline = transcript_lines[j]
            if pline.is_scene:
                break
            if pline.speaker:
                prev_speaker = pline.speaker
                break
        for j in range(i + 1, min(i + 5, len(transcript_lines))):
            nline = transcript_lines[j]
            if nline.is_scene:
                break
            if nline.speaker:
                next_speaker = nline.speaker
                break

        # If same speaker before and after, and they're different, this line is likely
        # the other speaker (dialogue alternation)
        if prev_speaker and next_speaker and prev_speaker == next_speaker:
            # Look for the other active speaker in this scene
            scene_speakers: set[str] = set()
            for j in range(max(0, i - 10), min(len(transcript_lines), i + 10)):
                sl = transcript_lines[j]
                if sl.is_scene and j != i:
                    if j < i:
                        scene_speakers.clear()
                    else:
                        break
                if sl.speaker:
                    scene_speakers.add(sl.speaker)

            other_speakers = scene_speakers - {prev_speaker}
            if len(other_speakers) == 1:
                tline.speaker = other_speakers.pop()
                tline.source = "rule"
                tline.confidence = 0.4
                continue

    return transcript_lines


# ---------------------------------------------------------------------------
# Vision-based speaker identification (batched)
# ---------------------------------------------------------------------------


def extract_frame(video_path: str, timestamp_secs: float, output_path: str) -> bool:
    """Extract a single video frame at the given timestamp."""
    ffmpeg = find_tool("ffmpeg")
    result = subprocess.run(
        [
            ffmpeg, "-y", "-ss", str(timestamp_secs),
            "-i", video_path, "-frames:v", "1",
            "-q:v", "2", output_path,
        ],
        capture_output=True, text=True,
    )
    return result.returncode == 0


def image_hash(path: str) -> str:
    """Simple average hash for frame deduplication."""
    data = Path(path).read_bytes()
    return hashlib.md5(data).hexdigest()


def get_context_lines(
    transcript_lines: list[TranscriptLine], batch_indices: list[int], context_n: int = 4,
) -> tuple[list[str], list[str]]:
    """Get labeled lines before and after a batch for conversational context."""
    first_idx = batch_indices[0]
    last_idx = batch_indices[-1]

    before: list[str] = []
    for j in range(first_idx - 1, max(first_idx - 20, -1), -1):
        line = transcript_lines[j]
        if line.speaker and not line.is_scene:
            before.insert(0, f"{line.speaker}: {line.text}")
            if len(before) >= context_n:
                break
        elif line.is_scene:
            before.insert(0, line.original)
            break

    after: list[str] = []
    for j in range(last_idx + 1, min(last_idx + 20, len(transcript_lines))):
        line = transcript_lines[j]
        if line.speaker and not line.is_scene:
            after.append(f"{line.speaker}: {line.text}")
            if len(after) >= context_n:
                break
        elif line.is_scene:
            after.append(line.original)
            break

    return before, after


def identify_batch_vision(
    frame_paths: list[str],
    dialogue_lines: list[str],
    context_before: list[str],
    context_after: list[str],
    episode_context: str,
    known_speakers: list[str],
) -> list[str]:
    """Send a batch of frames + dialogue to Claude Vision, get speaker labels."""
    try:
        import base64

        import anthropic
    except ImportError:
        print("Error: pip install anthropic  (required for --vision)", file=sys.stderr)
        sys.exit(1)

    client = anthropic.Anthropic()

    content: list[dict] = []

    # Add frames (skip any that vanished between extraction and API call)
    valid_frames = [fp for fp in frame_paths if Path(fp).exists()]
    if not valid_frames:
        return ["???"] * len(dialogue_lines)

    for fp in valid_frames:
        img_data = Path(fp).read_bytes()
        b64 = base64.standard_b64encode(img_data).decode("utf-8")
        suffix = Path(fp).suffix.lower()
        media = "image/jpeg" if suffix in (".jpg", ".jpeg") else "image/png"
        content.append({
            "type": "image",
            "source": {"type": "base64", "media_type": media, "data": b64},
        })

    # Build dialogue listing
    numbered = "\n".join(f"{i+1}. {line}" for i, line in enumerate(dialogue_lines))

    ctx_before_str = "\n".join(context_before) if context_before else "(scene start)"
    ctx_after_str = "\n".join(context_after) if context_after else "(scene end)"
    speakers_str = ", ".join(known_speakers) if known_speakers else "unknown"

    content.append({
        "type": "text",
        "text": (
            f"Episode: {episode_context}\n"
            f"Known characters in this scene: {speakers_str}\n\n"
            f"PRECEDING DIALOGUE:\n{ctx_before_str}\n\n"
            f"UNLABELED LINES (identify the speaker for each):\n{numbered}\n\n"
            f"FOLLOWING DIALOGUE:\n{ctx_after_str}\n\n"
            "The frames above are from this scene. Look at who is on screen, "
            "mouth movements, body language, and conversational context to determine "
            "who speaks each numbered line.\n\n"
            "Respond with ONLY a numbered list matching the lines above:\n"
            "1. CharacterName\n2. CharacterName\n...\n"
            "Use ??? if you cannot determine a speaker."
        ),
    })

    # Retry with exponential backoff for rate limits
    max_retries = 5
    for attempt in range(max_retries):
        try:
            response = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=500,
                messages=[{"role": "user", "content": content}],
            )
            break
        except Exception as e:
            if "rate_limit" in str(e).lower() or "429" in str(e):
                wait = 2 ** attempt * 10  # 10s, 20s, 40s, 80s, 160s
                print(f"    Rate limited, waiting {wait}s...", file=sys.stderr)
                time.sleep(wait)
            else:
                raise
    else:
        print("    Max retries exceeded, skipping batch", file=sys.stderr)
        return ["???"] * len(dialogue_lines)

    # Parse response into list of speaker names
    response_text = response.content[0].text.strip()
    speakers: list[str] = []
    for line in response_text.split("\n"):
        line = line.strip()
        m = re.match(r"\d+\.\s*(.+)", line)
        if m:
            speakers.append(m.group(1).strip())

    # Pad if response is short
    while len(speakers) < len(dialogue_lines):
        speakers.append("???")

    return speakers


def group_unlabeled_batches(
    transcript_lines: list[TranscriptLine],
) -> list[list[int]]:
    """Group consecutive unlabeled lines into scene-bounded batches."""
    batches: list[list[int]] = []
    current_batch: list[int] = []

    for i, tline in enumerate(transcript_lines):
        if tline.is_scene:
            # Scene break: flush current batch
            if current_batch:
                batches.append(current_batch)
                current_batch = []
            continue

        if not tline.speaker and not tline.is_scene and tline.matched_srt:
            current_batch.append(i)
            if len(current_batch) >= VISION_BATCH_SIZE:
                batches.append(current_batch)
                current_batch = []
        else:
            if current_batch:
                batches.append(current_batch)
                current_batch = []

    if current_batch:
        batches.append(current_batch)

    return batches


def checkpoint_path_for(transcript_path: Path) -> Path:
    """Return the checkpoint file path for a given transcript."""
    return transcript_path.with_suffix(".checkpoint.json")


def save_checkpoint(
    ckpt_path: Path,
    transcript_lines: list[TranscriptLine],
    completed_batches: int,
) -> None:
    """Save attribution progress to checkpoint file."""
    data = {
        "completed_batches": completed_batches,
        "attributions": {
            str(t.line_num): {
                "speaker": t.speaker,
                "source": t.source,
                "confidence": t.confidence,
            }
            for t in transcript_lines
            if t.source in ("vision", "rule", "sdh")
        },
    }
    ckpt_path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_checkpoint(
    ckpt_path: Path,
    transcript_lines: list[TranscriptLine],
) -> int:
    """Restore attribution progress from checkpoint. Returns completed batch count."""
    if not ckpt_path.exists():
        return 0

    data = json.loads(ckpt_path.read_text(encoding="utf-8"))
    attributions = data.get("attributions", {})

    restored = 0
    for tline in transcript_lines:
        key = str(tline.line_num)
        if key in attributions:
            attr = attributions[key]
            tline.speaker = attr["speaker"]
            tline.source = attr["source"]
            tline.confidence = attr["confidence"]
            if attr["source"] == "vision":
                tline.vision_speaker = attr["speaker"]
            restored += 1

    completed = data.get("completed_batches", 0)
    if restored:
        print(f"  Restored {restored} attributions from checkpoint (batch {completed})")
    return completed


def run_vision_attribution(
    transcript_lines: list[TranscriptLine],
    video_path: str,
    episode_context: str,
    transcript_path: Path | None = None,
) -> list[TranscriptLine]:
    """Run batched Vision-based speaker ID on unlabeled lines with checkpointing."""
    # Collect known speakers for context
    known_speakers = sorted({
        t.speaker for t in transcript_lines if t.speaker
    })

    # Load checkpoint if available
    ckpt_path = checkpoint_path_for(transcript_path) if transcript_path else None
    start_batch = 0
    if ckpt_path:
        start_batch = load_checkpoint(ckpt_path, transcript_lines)

    batches = group_unlabeled_batches(transcript_lines)
    total_batches = len(batches)
    total_lines = sum(len(b) for b in batches)

    if not batches:
        print("  No unlabeled lines with timestamps to process.")
        return transcript_lines

    remaining = total_batches - start_batch
    print(f"  {total_lines} lines in {total_batches} batches ({remaining} remaining)")

    with tempfile.TemporaryDirectory() as tmpdir:
        for batch_num, batch_indices in enumerate(batches, 1):
            if batch_num <= start_batch:
                continue

            batch_lines = [transcript_lines[i] for i in batch_indices]
            # Skip lines already attributed (from checkpoint)
            unanswered = [i for i, t in zip(batch_indices, batch_lines) if not t.speaker]
            if not unanswered:
                continue

            dialogue = [transcript_lines[i].text for i in unanswered]

            timestamps = [
                transcript_lines[i].matched_srt.start_seconds
                for i in unanswered
                if transcript_lines[i].matched_srt
            ]
            if not timestamps:
                continue

            if len(timestamps) <= MAX_FRAMES_PER_BATCH:
                sample_times = timestamps
            else:
                step = len(timestamps) / MAX_FRAMES_PER_BATCH
                sample_times = [timestamps[int(i * step)] for i in range(MAX_FRAMES_PER_BATCH)]

            frame_paths: list[str] = []
            seen_hashes: set[str] = set()
            for ts in sample_times:
                fpath = str(Path(tmpdir) / f"batch{batch_num}_{ts:.3f}.jpg")
                if not extract_frame(video_path, ts, fpath):
                    continue
                if not Path(fpath).exists() or Path(fpath).stat().st_size < 100:
                    continue
                h = image_hash(fpath)
                if h in seen_hashes:
                    Path(fpath).unlink(missing_ok=True)
                    continue
                seen_hashes.add(h)
                frame_paths.append(fpath)

            if not frame_paths:
                continue

            ctx_before, ctx_after = get_context_lines(transcript_lines, unanswered)

            print(
                f"  Batch {batch_num}/{total_batches}: "
                f"{len(dialogue)} lines, {len(frame_paths)} frames"
            )

            speakers = identify_batch_vision(
                frame_paths, dialogue, ctx_before, ctx_after,
                episode_context, known_speakers,
            )

            for idx, speaker in zip(unanswered, speakers):
                if speaker and speaker != "???":
                    transcript_lines[idx].speaker = speaker
                    transcript_lines[idx].vision_speaker = speaker
                    transcript_lines[idx].source = "vision"
                    transcript_lines[idx].confidence = 0.7

            # Checkpoint after each batch
            if ckpt_path:
                save_checkpoint(ckpt_path, transcript_lines, batch_num)

            # Clean up frame files for this batch
            for fp in frame_paths:
                Path(fp).unlink(missing_ok=True)

    return transcript_lines


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def format_output(transcript_lines: list[TranscriptLine], original_text: str) -> str:
    """Rebuild transcript with speaker labels applied."""
    merged = merge_subtitle_lines(original_text)
    output_lines: list[str] = []

    attribution: dict[str, str] = {}
    for tline in transcript_lines:
        if tline.speaker and not SPEAKER_RE.match(tline.original):
            attribution[tline.text] = tline.speaker

    for line in merged.split("\n"):
        stripped = line.strip()
        if not stripped:
            output_lines.append("")
            continue

        if not SPEAKER_RE.match(stripped) and not stripped.startswith("["):
            best_key = ""
            best_ratio = 0.0
            for key in attribution:
                ratio = SequenceMatcher(None, stripped.lower(), key.lower()).ratio()
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_key = key
            if best_ratio >= 0.8 and best_key:
                speaker = attribution[best_key]
                output_lines.append(f"{speaker}:  {stripped}")
                continue

        output_lines.append(stripped)

    return "\n".join(output_lines)


def print_stats(transcript_lines: list[TranscriptLine]) -> None:
    """Print summary of attribution results."""
    total = sum(1 for t in transcript_lines if not t.is_scene)
    labeled = sum(1 for t in transcript_lines if t.speaker and not t.is_scene)
    by_source = {}
    for t in transcript_lines:
        if t.source:
            by_source[t.source] = by_source.get(t.source, 0) + 1
    unlabeled = total - labeled

    print("\nAttribution summary:")
    print(f"  Total dialogue lines: {total}")
    for src in ("sdh", "rule", "vision"):
        if src in by_source:
            print(f"  From {src:17s}: {by_source[src]}")
    already = labeled - sum(by_source.values())
    if already > 0:
        print(f"  Already labeled:      {already}")
    print(f"  Still unlabeled:      {unlabeled}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("transcript", help="Path to transcript .txt file")
    parser.add_argument("--video", help="Path to video file (MKV/MP4)")
    parser.add_argument("--srt", help="Path to pre-extracted SDH SRT file")
    parser.add_argument("--scan-only", action="store_true", help="Just list subtitle tracks")
    parser.add_argument("--vision", action="store_true", help="Use Claude Vision for unlabeled lines")
    parser.add_argument("--write", action="store_true", help="Write changes to transcript file")
    parser.add_argument(
        "--context", default="Adventure Time",
        help="Episode context for Vision prompts (e.g. 'Adventure Time S08E21')",
    )
    args = parser.parse_args()

    transcript_path = Path(args.transcript)
    if not transcript_path.exists():
        print(f"Error: {transcript_path} not found", file=sys.stderr)
        sys.exit(1)

    # Scan-only mode
    if args.scan_only:
        if not args.video:
            print("Error: --scan-only requires --video", file=sys.stderr)
            sys.exit(1)
        tracks = scan_subtitle_tracks(args.video)
        if not tracks:
            print("No subtitle tracks found.")
            return
        print(f"Subtitle tracks in {args.video}:")
        for t in tracks:
            text_marker = " [TEXT]" if t.is_text else " [BITMAP]"
            print(f"  #{t.index}: {t.codec}{text_marker}  lang={t.language}  title={t.title!r}")
        return

    # Read and merge transcript
    original_text = transcript_path.read_text(encoding="utf-8")
    merged_text = merge_subtitle_lines(original_text)
    transcript_lines = parse_transcript(merged_text)

    # Get SRT entries
    srt_entries: list[SrtEntry] = []
    srt_temp = None

    if args.srt:
        srt_entries = parse_srt(args.srt)
        print(f"Loaded {len(srt_entries)} SRT entries from {args.srt}")
    elif args.video:
        tracks = scan_subtitle_tracks(args.video)
        text_tracks = [t for t in tracks if t.is_text]

        if text_tracks:
            sdh_tracks = [
                t for t in text_tracks
                if "sdh" in t.title.lower() or "cc" in t.title.lower()
            ]
            track = sdh_tracks[0] if sdh_tracks else text_tracks[0]

            srt_temp = tempfile.NamedTemporaryFile(suffix=".srt", delete=False)
            srt_temp.close()
            print(f"Extracting subtitle track #{track.index} ({track.codec}, {track.title!r})...")
            if extract_subtitle_track(args.video, track.index, srt_temp.name):
                srt_entries = parse_srt(srt_temp.name)
                print(f"Parsed {len(srt_entries)} SRT entries")
            else:
                print("Warning: subtitle extraction failed", file=sys.stderr)
        else:
            print("No text-based subtitle tracks found in video.")
            if not args.vision:
                print("Use --vision to attempt frame-based identification.")

    # Match transcript to SRT (gives us timestamps even if no SDH speaker labels)
    if srt_entries:
        transcript_lines = match_transcript_to_srt(transcript_lines, srt_entries)
        speakers_found = sum(
            1 for t in transcript_lines
            if t.source == "sdh"
        )
        print(f"Matched {speakers_found} speaker labels from SDH")

    # Rule-based inference
    print("Running rule-based speaker inference...")
    transcript_lines = infer_speakers_rules(transcript_lines)
    rule_count = sum(1 for t in transcript_lines if t.source == "rule")
    print(f"Inferred {rule_count} speakers from rules")

    # Vision fallback (batched)
    if args.vision and args.video:
        unlabeled = [
            t for t in transcript_lines
            if not t.speaker and not t.is_scene and t.matched_srt
        ]
        if unlabeled:
            print(f"Running Vision AI on {len(unlabeled)} unlabeled lines...")
            transcript_lines = run_vision_attribution(
                transcript_lines, args.video, args.context, transcript_path,
            )

    # Second rule-based pass (uses Vision labels as context for turn-taking)
    pre_rules2 = sum(1 for t in transcript_lines if t.source == "rule")
    transcript_lines = infer_speakers_rules(transcript_lines)
    rules2 = sum(1 for t in transcript_lines if t.source == "rule") - pre_rules2
    if rules2:
        print(f"Second rule pass inferred {rules2} more speakers")

    # Output
    print_stats(transcript_lines)
    result = format_output(transcript_lines, original_text)

    if args.write:
        transcript_path.write_text(result, encoding="utf-8")
        print(f"\nWrote: {transcript_path}")
    else:
        print("\nPreview (first 40 lines of output):")
        for line in result.split("\n")[:40]:
            print(f"  {line}")
        print(f"\n  ... ({len(result.splitlines())} total lines)")
        print("\nUse --write to apply changes.")

    # Cleanup temp file
    if srt_temp:
        Path(srt_temp.name).unlink(missing_ok=True)


if __name__ == "__main__":
    main()
