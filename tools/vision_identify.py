#!/usr/bin/env python3
"""Speaker identification for diarization clusters via video clips or Vision API.

Modes:
  --clips   Interactive: extract clips per cluster, play in VLC, prompt for ID
  (default) Automated: extract frames, send to Claude Sonnet Vision API

Works independently of transcript files — uses only whisperX diarization
output + video.

Requirements:
    ffmpeg on PATH
    VLC installed (for --clips mode)
    pip install anthropic + ANTHROPIC_API_KEY env var (for Vision mode)

Usage:
    # Interactive clip review (recommended)
    python tools/vision_identify.py --clips --episode FC.S02E02 --video "D:\\Shows\\file.mkv"

    # Vision API identification
    python tools/vision_identify.py --episode FC.S02E02 --video "D:\\Shows\\file.mkv"

    # Batch mode (either mode)
    python tools/vision_identify.py --clips --series fc --season 2 --video-dir "D:\\Shows"
"""

from __future__ import annotations

import argparse
import base64
import json
import re
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

# Reuse frame extraction + dedup from extract_speakers
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
from extract_speakers import extract_frame, image_hash

from tools.diarize.vlc import (
    extract_clip,
    find_vlc,
    play_clips_vlc,
    stop_vlc,
)

# --- Paths -----------------------------------------------------------

SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
DIARIZATION_DIR = PROJECT_DIR / "diarization"
LABELS_DIR = DIARIZATION_DIR / "labels"
PROFILES_DIR = DIARIZATION_DIR / "voice_profiles"

# --- Config -----------------------------------------------------------

MODEL = "claude-sonnet-4-5-20250929"
DEFAULT_SAMPLES = 5
MIN_SEGMENTS = 3
MAX_CLUSTERS_PER_BATCH = 8
MAX_DIALOGUE_LINES = 10

CLIP_SAMPLES = 7        # clips to play per cluster in interactive mode
CLIP_EXPAND = 5         # additional clips per "more" request

EPISODE_RE = re.compile(r"S(\d{2})E(\d{2,3})")
VIDEO_EXTS = {".mkv", ".mp4", ".avi", ".webm"}

SERIES_NAMES = {
    "AT": "Adventure Time",
    "DL": "Adventure Time: Distant Lands",
    "FC": "Adventure Time: Fionna & Cake",
}

# Series-specific character context for the Vision prompt.
SERIES_CHARACTERS = {
    "FC": (
        "Main cast: Fionna (blonde human girl, bunny hat), "
        "Cake (shapeshifting cat), Simon Petrikov (glasses, grey hair, human), "
        "Marshall Lee (vampire, dark hair), Gary Prince (pink gumball person), "
        "Flame Prince / DJ Flame (fire elemental boy), Scarab (insectoid villain), "
        "Prismo (2D shadow being in a time room).\n"
        "Recurring: Huntress Wizard / Hunter (green, antlers), "
        "Ice Queen (blue skin, crown), Witch Wizard (green wizard), "
        "Ellis P (Fionna's adoptive parent, elephant-like), "
        "Cosmic Owl (large owl), Spirit of the Forest, "
        "Karmic Worm (worm creature), Big Destiny.\n"
        "NOTE: This is a different universe from Adventure Time. Characters like "
        "Finn, Jake, Princess Bubblegum, Marceline do NOT appear in Fionna & Cake. "
        "Do not use AT character names for FC characters."
    ),
    "DL": (
        "Main cast varies by episode: BMO, Marceline, Princess Bubblegum, "
        "Finn, Jake, Peppermint Butler.\n"
        "These are special episodes set in the Adventure Time universe."
    ),
    "AT": (
        "Main cast: Finn (human boy, white hat), Jake (yellow stretchy dog), "
        "Princess Bubblegum / PB (pink, crown), Marceline (vampire, dark hair), "
        "Ice King (blue, crown, beard), BMO (small green game console), "
        "Lumpy Space Princess / LSP (purple, floating).\n"
        "Recurring: Flame Princess (fire girl), Lady Rainicorn, Tree Trunks, "
        "Cinnamon Bun, Peppermint Butler, Lemongrab, Huntress Wizard."
    ),
}

# --- Data loading -----------------------------------------------------


def load_diarization(episode_id: str) -> dict:
    """Load v2 diarization JSON for an episode."""
    path = DIARIZATION_DIR / f"{episode_id}.json"
    if not path.exists():
        print(f"Error: No diarization data at {path}", file=sys.stderr)
        print("Run 'process --episode' first.", file=sys.stderr)
        sys.exit(1)
    return json.loads(path.read_text(encoding="utf-8"))


def group_by_speaker(data: dict) -> dict[str, list[dict]]:
    """Group diarization segments by speaker label."""
    groups: dict[str, list[dict]] = {}
    for seg in data.get("segments", []):
        spk = seg.get("speaker", "")
        if not spk:
            continue
        groups.setdefault(spk, []).append(seg)
    return groups


def pick_samples(segments: list[dict], n: int) -> list[dict]:
    """Pick N representative segments evenly spaced across time."""
    if len(segments) <= n:
        return list(segments)
    step = len(segments) / n
    return [segments[int(i * step)] for i in range(n)]


def pick_best_clips(segments: list[dict], n: int) -> list[dict]:
    """Pick N segments best suited for clip playback.

    Prefers longer segments with dialogue, evenly spread across time.
    """
    # Filter to segments with text and > 1s duration
    viable = [
        s for s in segments
        if s.get("text", "").strip() and (s["end"] - s["start"]) > 1.0
    ]
    if not viable:
        viable = segments  # fall back to all

    # Sort by duration descending, take top 2*n, then spread evenly
    by_dur = sorted(viable, key=lambda s: s["end"] - s["start"], reverse=True)
    pool = by_dur[:min(n * 2, len(by_dur))]

    # Re-sort by time and pick evenly spaced
    pool.sort(key=lambda s: s["start"])
    return pick_samples(pool, n)


def get_known_characters(series_code: str = "") -> list[str]:
    """Load character names from voice profile index, filtered by series."""
    index_path = PROFILES_DIR / "_index.json"
    if not index_path.exists():
        return []
    index = json.loads(index_path.read_text(encoding="utf-8"))
    profiles = index.get("profiles", {})

    if not series_code:
        return sorted(profiles.keys())

    filtered = []
    for name, meta in profiles.items():
        first = meta.get("first_episode", "")
        last = meta.get("last_episode", "")
        if last and f"{series_code}." > last:
            continue
        if first and f"{series_code}." < first.split(".")[0] + ".":
            continue
        filtered.append(name)

    return sorted(filtered)


def series_context(episode_id: str) -> tuple[str, str, int]:
    """Extract series name, code, and season from episode ID."""
    code = episode_id.split(".")[0]
    name = SERIES_NAMES.get(code, "Adventure Time")
    m = EPISODE_RE.search(episode_id)
    season = int(m.group(1)) if m else 0
    return name, code, season


def review_episode_interactive(
    episode_id: str,
    video_path: str,
    min_segs: int,
    apply: bool,
    audio_track: int | None = None,
    speakers: list[str] | None = None,
) -> dict[str, str]:
    """Interactive clip review: play clips per cluster, prompt user for ID."""
    vlc_path = find_vlc()
    if not vlc_path:
        print("Error: VLC not found. Install VLC or add it to PATH.", file=sys.stderr)
        sys.exit(1)

    print(f"\n{'=' * 60}")
    print(f"Episode: {episode_id}")
    print(f"Video:   {video_path}")
    print("Mode:    Interactive clip review (VLC)")
    print()

    data = load_diarization(episode_id)
    groups = group_by_speaker(data)
    title = data.get("title", "")

    if title:
        print(f"Title:   {title}")

    # Sort clusters by segment count (largest first — main characters)
    sorted_speakers = sorted(groups.keys(), key=lambda s: len(groups[s]), reverse=True)

    if speakers:
        # Target specific speakers — no min-segs filtering
        eligible = [s for s in sorted_speakers if s in speakers]
        missing = [s for s in speakers if s not in groups]
        if missing:
            print(f"  Warning: speakers not found: {', '.join(missing)}")
        print(
            f"  {len(groups)} clusters, reviewing {len(eligible)} targeted\n"
        )
    else:
        eligible = [s for s in sorted_speakers if len(groups[s]) >= min_segs]
        small = [s for s in sorted_speakers if len(groups[s]) < min_segs]
        print(
            f"  {len(groups)} clusters, {len(eligible)} eligible "
            f"(>= {min_segs} segs), {len(small)} too small\n"
        )

    speaker_map: dict[str, str] = {}
    skipped: list[str] = [] if speakers else [
        s for s in sorted_speakers if len(groups[s]) < min_segs
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        for i, spk in enumerate(eligible):
            segs = groups[spk]
            total_dur = sum(s["end"] - s["start"] for s in segs)
            samples = pick_best_clips(segs, CLIP_SAMPLES)

            # Show cluster info
            print(f"--- [{i + 1}/{len(eligible)}] {spk} "
                  f"({len(segs)} segs, {total_dur:.1f}s) ---")

            # Show dialogue samples
            dialogue_segs = pick_samples(segs, MAX_DIALOGUE_LINES)
            for seg in dialogue_segs:
                txt = seg.get("text", "").strip()
                if txt:
                    print(f'  "{txt}"')

            # Clips are extracted lazily — only when requested
            clip_paths: list[str] = []
            used_segs: set[int] = set()
            clip_counter = 0
            vlc_proc: subprocess.Popen | None = None

            def _extract_clips(seg_list, label=""):
                """Extract clips from segments, return new clip paths."""
                nonlocal clip_counter
                new_clips: list[str] = []
                for seg in seg_list:
                    clip_path = str(Path(tmpdir) / f"{spk}_clip{clip_counter}.mp4")
                    clip_counter += 1
                    if extract_clip(video_path, seg["start"], seg["end"], clip_path, audio_track):
                        new_clips.append(clip_path)
                    for idx, s in enumerate(segs):
                        if s["start"] == seg["start"] and s["end"] == seg["end"]:
                            used_segs.add(idx)
                            break
                return new_clips

            # Auto-extract and play clips (non-blocking VLC)
            print(f"  Extracting {len(samples)} clips...")
            clip_paths = _extract_clips(samples)
            if clip_paths:
                print(f"  Playing {len(clip_paths)} clips...")
                vlc_proc = play_clips_vlc(vlc_path, clip_paths)
            else:
                print("  (no clips could be extracted)")

            while True:
                print()
                answer = input(
                    f"  {spk} = ? (name / skip / more / replay / quit): "
                ).strip()

                if answer.lower() == "":
                    continue
                elif answer.lower() in ("replay", "r"):
                    stop_vlc(vlc_proc)
                    if clip_paths:
                        print(f"  Replaying {len(clip_paths)} clips...")
                        vlc_proc = play_clips_vlc(vlc_path, clip_paths)
                    else:
                        print("  (no clips yet — use 'more' first)")
                    continue
                elif answer.lower() in ("more", "m"):
                    stop_vlc(vlc_proc)
                    # Find segments not yet clipped
                    remaining = [
                        s for idx, s in enumerate(segs) if idx not in used_segs
                    ]
                    if not remaining:
                        print("  (no more segments available)")
                        continue
                    extra = pick_best_clips(remaining, CLIP_EXPAND)
                    new_clips: list[str] = []
                    for seg in extra:
                        clip_path = str(Path(tmpdir) / f"{spk}_clip{clip_counter}.mp4")
                        clip_counter += 1
                        if extract_clip(video_path, seg["start"], seg["end"], clip_path, audio_track):
                            new_clips.append(clip_path)
                        for idx, s in enumerate(segs):
                            if s["start"] == seg["start"] and s["end"] == seg["end"]:
                                used_segs.add(idx)
                                break
                    if new_clips:
                        clip_paths.extend(new_clips)
                        print(f"  Playing {len(new_clips)} more clips "
                              f"({len(used_segs)}/{len(segs)} segments used)...")
                        vlc_proc = play_clips_vlc(vlc_path, new_clips)
                    else:
                        print("  (no more clips could be extracted)")
                    continue
                # User submitted a name, skip, or quit — stop VLC
                stop_vlc(vlc_proc)
                break

            if answer.lower() in ("quit", "q"):
                print("  Quitting review.")
                break
            elif answer.lower() in ("skip", "s"):
                skipped.append(spk)
                print("  -> skipped")
            else:
                speaker_map[spk] = answer
                print(f"  -> {answer}")

            print()

    # Summary
    summary_lines: list[str] = []
    summary_lines.append(f"Summary for {episode_id}:")
    for spk in sorted_speakers:
        seg_count = len(groups[spk])
        if spk in speaker_map:
            summary_lines.append(f"  {spk}: -> {speaker_map[spk]} ({seg_count} segs)")
        elif spk in skipped:
            summary_lines.append(f"  {spk}: (skipped, {seg_count} segs)")
        else:
            summary_lines.append(f"  {spk}: (not reviewed, {seg_count} segs)")
            skipped.append(spk)

    mapped = len(speaker_map)
    total = len(groups)
    summary_lines.append(f"\n  Mapped {mapped}/{total} clusters")

    if apply and speaker_map:
        write_labels(episode_id, data, speaker_map, skipped)
    elif speaker_map and not apply:
        summary_lines.append("\n  (preview only — use --apply to write label file)")

    summary = "\n".join(summary_lines)
    print(f"\n{'=' * 60}")
    print(summary)

    # Copy summary to clipboard (Windows)
    try:
        subprocess.run(
            ["clip.exe"], input=summary.encode("utf-8"),
            timeout=5, check=True,
        )
        print("\n  (copied to clipboard)")
    except Exception:
        pass

    return speaker_map


# --- Vision API -------------------------------------------------------


def identify_clusters(
    clusters_data: list[dict],
    episode_id: str,
    known_characters: list[str],
) -> dict[str, str]:
    """Send cluster frames + dialogue to Claude Vision for identification."""
    try:
        import anthropic
    except ImportError:
        print("Error: pip install anthropic", file=sys.stderr)
        sys.exit(1)

    client = anthropic.Anthropic()
    content: list[dict] = []

    for cluster in clusters_data:
        spk = cluster["speaker"]
        content.append({
            "type": "text",
            "text": (
                f"\n--- {spk} ({cluster['seg_count']} segments, "
                f"{cluster['duration']:.1f}s total) ---"
            ),
        })

        for frame_path in cluster["frames"]:
            img_data = Path(frame_path).read_bytes()
            b64 = base64.standard_b64encode(img_data).decode("utf-8")
            content.append({
                "type": "image",
                "source": {"type": "base64", "media_type": "image/jpeg", "data": b64},
            })

        dialogue = "\n".join(f'  "{line}"' for line in cluster["dialogue"])
        content.append({"type": "text", "text": f"Dialogue:\n{dialogue}"})

    name, code, season = series_context(episode_id)
    title = clusters_data[0].get("title", "") if clusters_data else ""
    title_str = f' — "{title}"' if title else ""
    series_chars = SERIES_CHARACTERS.get(code, "")
    profile_names = ", ".join(known_characters[:40]) if known_characters else ""

    prompt_parts = [
        f"\nEpisode: {episode_id}{title_str} ({name}, Season {season})\n",
    ]
    if series_chars:
        prompt_parts.append(f"Character guide for this series:\n{series_chars}\n")
    if profile_names:
        prompt_parts.append(
            f"Voice profiles available (for reference only, do NOT constrain "
            f"your answers to this list): {profile_names}\n"
        )
    prompt_parts.append(
        "\nFor each SPEAKER_XX above, identify the character based on their "
        "visual appearance in the frames and their dialogue.\n\n"
        "Notes:\n"
        "- This is an animated show — focus on character design (hair, color, "
        "body shape, clothing) rather than subtle facial movements\n"
        "- The speaking character may be off-screen (reverse shot showing "
        "the listener) — use dialogue content to help identify\n"
        "- If frames show DIFFERENT characters across segments, the cluster "
        "contains multiple speakers — mark as MIXED\n"
        "- Use the character's most common name from the series guide above. "
        "For minor/background characters, use a brief description "
        "(e.g. 'Guard', 'Bartender')\n\n"
        "Respond with EXACTLY one line per speaker:\n"
        "SPEAKER_XX = CharacterName\n"
        "SPEAKER_XX = MIXED\n"
        "SPEAKER_XX = UNKNOWN\n"
        "Use ? suffix if uncertain: SPEAKER_XX = Fionna?\n"
    )

    content.append({"type": "text", "text": "\n".join(prompt_parts)})

    max_retries = 5
    response = None
    for attempt in range(max_retries):
        try:
            response = client.messages.create(
                model=MODEL,
                max_tokens=1000,
                messages=[{"role": "user", "content": content}],
            )
            break
        except Exception as e:
            if "rate_limit" in str(e).lower() or "429" in str(e):
                wait = 2 ** attempt * 10
                print(f"    Rate limited, waiting {wait}s...", file=sys.stderr)
                time.sleep(wait)
            else:
                raise

    if response is None:
        print("    Max retries exceeded, skipping batch", file=sys.stderr)
        return {}

    usage = response.usage
    print(
        f"    Tokens: {usage.input_tokens:,} in, {usage.output_tokens:,} out"
        f"  (${usage.input_tokens * 3 / 1_000_000 + usage.output_tokens * 15 / 1_000_000:.3f})"
    )

    result: dict[str, str] = {}
    text = response.content[0].text
    for line in text.strip().split("\n"):
        line = line.strip()
        if "=" in line and "SPEAKER_" in line:
            parts = line.split("=", 1)
            speaker = parts[0].strip()
            character = parts[1].strip()
            result[speaker] = character

    return result


# --- Vision episode processing ----------------------------------------


def process_episode_vision(
    episode_id: str,
    video_path: str,
    n_samples: int,
    min_segs: int,
    apply: bool,
) -> dict[str, str]:
    """Process a single episode via Vision API."""
    print(f"\n{'=' * 60}")
    print(f"Episode: {episode_id}")
    print(f"Video:   {video_path}")
    print(f"Mode:    Vision API ({MODEL})")

    data = load_diarization(episode_id)
    groups = group_by_speaker(data)
    _, code, _ = series_context(episode_id)
    known_chars = get_known_characters(code)
    title = data.get("title", "")

    eligible: dict[str, list[dict]] = {}
    small_skipped: list[str] = []

    for spk in sorted(groups):
        if len(groups[spk]) >= min_segs:
            eligible[spk] = groups[spk]
        else:
            small_skipped.append(spk)

    print(
        f"  {len(groups)} clusters total, {len(eligible)} eligible "
        f"(>= {min_segs} segs), {len(small_skipped)} too small"
    )

    if not eligible:
        print("  No eligible clusters.")
        return {}

    all_results: dict[str, str] = {}

    with tempfile.TemporaryDirectory() as tmpdir:
        clusters_data: list[dict] = []

        for spk, segs in eligible.items():
            frame_samples = pick_samples(segs, n_samples)
            frames: list[str] = []
            seen_hashes: set[str] = set()

            for i, seg in enumerate(frame_samples):
                mid = (seg["start"] + seg["end"]) / 2
                fpath = str(Path(tmpdir) / f"{spk}_{i}.jpg")

                if not extract_frame(video_path, mid, fpath):
                    continue
                if not Path(fpath).exists() or Path(fpath).stat().st_size < 100:
                    continue

                h = image_hash(fpath)
                if h in seen_hashes:
                    Path(fpath).unlink(missing_ok=True)
                    continue
                seen_hashes.add(h)
                frames.append(fpath)

            if not frames:
                print(f"    {spk}: no frames extracted, skipping")
                small_skipped.append(spk)
                continue

            dialogue_samples = pick_samples(segs, MAX_DIALOGUE_LINES)
            dialogue = [
                s.get("text", "").strip()
                for s in dialogue_samples
                if s.get("text", "").strip()
            ]

            total_dur = sum(s["end"] - s["start"] for s in segs)
            clusters_data.append({
                "speaker": spk,
                "frames": frames,
                "dialogue": dialogue,
                "seg_count": len(segs),
                "duration": total_dur,
                "title": title,
            })

        for batch_start in range(0, len(clusters_data), MAX_CLUSTERS_PER_BATCH):
            batch = clusters_data[batch_start:batch_start + MAX_CLUSTERS_PER_BATCH]
            speakers = ", ".join(c["speaker"] for c in batch)
            n_frames = sum(len(c["frames"]) for c in batch)
            print(f"  Vision batch: {speakers} ({n_frames} frames)")

            results = identify_clusters(batch, episode_id, known_chars)
            all_results.update(results)

            if batch_start + MAX_CLUSTERS_PER_BATCH < len(clusters_data):
                time.sleep(2)

    print("\n  Results:")
    speaker_map: dict[str, str] = {}
    skipped: list[str] = list(small_skipped)

    for spk in sorted(groups):
        seg_count = len(groups[spk])
        char = all_results.get(spk, "")

        if spk in small_skipped:
            print(f"    {spk}: (skipped, {seg_count} segs)")
            continue

        if char in ("MIXED", "UNKNOWN", "") or char.endswith("?"):
            label = char or "no result"
            print(f"    {spk}: {label} ({seg_count} segs)")
            skipped.append(spk)
        else:
            print(f"    {spk}: -> {char} ({seg_count} segs)")
            speaker_map[spk] = char

    mapped = len(speaker_map)
    total = len(groups)
    print(f"\n  Mapped {mapped}/{total} clusters")

    if apply and speaker_map:
        write_labels(episode_id, data, speaker_map, skipped, source="vision")
    elif apply:
        print("  Nothing to apply.")

    return all_results


# --- Label I/O -------------------------------------------------------


def write_labels(
    episode_id: str,
    data: dict,
    speaker_map: dict[str, str],
    skipped: list[str],
    source: str = "clips",
) -> None:
    """Write or merge into label file (additive)."""
    LABELS_DIR.mkdir(parents=True, exist_ok=True)
    label_path = LABELS_DIR / f"{episode_id}.json"

    if label_path.exists():
        existing = json.loads(label_path.read_text(encoding="utf-8"))
        existing_map = existing.get("speaker_map", {})
        existing_skipped = set(existing.get("skipped", []))

        for spk, char in speaker_map.items():
            existing_map[spk] = char
            existing_skipped.discard(spk)

        for spk in skipped:
            if spk not in existing_map:
                existing_skipped.add(spk)

        existing["speaker_map"] = existing_map
        existing["skipped"] = sorted(existing_skipped)
        label_data = existing
    else:
        label_data = {
            "episode_id": episode_id,
            "title": data.get("title", ""),
            "season": data.get("season", 0),
            "speaker_map": speaker_map,
            "skipped": sorted(skipped),
            "source": source,
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"),
        }

    label_path.write_text(
        json.dumps(label_data, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"  Wrote {label_path}")


# --- Batch helpers ----------------------------------------------------


def find_video(video_dir: Path, episode_id: str) -> Path | None:
    """Find video file matching episode ID in a directory."""
    m = EPISODE_RE.search(episode_id)
    if not m:
        return None

    season_ep = f"S{m.group(1)}E{m.group(2)}"

    for ext in VIDEO_EXTS:
        for match in video_dir.glob(f"*{season_ep}*{ext}"):
            return match
        for match in video_dir.glob(f"*{season_ep.lower()}*{ext}"):
            return match

    return None


def discover_episodes(series: str, season: int) -> list[str]:
    """Find episode IDs with diarization data for a series/season."""
    prefix = f"{series.upper()}.S{season:02d}"
    episodes = []

    for path in sorted(DIARIZATION_DIR.glob(f"{prefix}*.json")):
        name = path.stem
        if name == "validation_progress":
            continue
        episodes.append(name)

    return episodes


# --- CLI --------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Speaker identification for diarization clusters",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            '  %(prog)s --clips --episode FC.S02E02 --video "D:\\Shows\\file.mkv"\n'
            '  %(prog)s --clips --episode FC.S02E02 --video "D:\\Shows\\file.mkv" --apply\n'
            '  %(prog)s --clips --episode FC.S02E02 --video "D:\\Shows\\file.mkv" --speakers SPEAKER_04 SPEAKER_13\n'
            '  %(prog)s --episode FC.S02E02 --video "D:\\Shows\\file.mkv"\n'
            '  %(prog)s --series fc --season 2 --video-dir "D:\\Shows"\n'
        ),
    )
    parser.add_argument(
        "--clips", action="store_true",
        help="Interactive mode: play clips in VLC, prompt for IDs",
    )
    parser.add_argument("--episode", help="Single episode ID (e.g. FC.S02E02)")
    parser.add_argument("--video", help="Path to video file (single episode mode)")
    parser.add_argument(
        "--series", choices=["at", "dl", "fc"],
        help="Series for batch mode",
    )
    parser.add_argument("--season", type=int, help="Season number for batch mode")
    parser.add_argument("--video-dir", help="Video directory for batch mode")
    parser.add_argument(
        "--samples", type=int, default=DEFAULT_SAMPLES,
        help=f"Frames per cluster, Vision mode (default: {DEFAULT_SAMPLES})",
    )
    parser.add_argument(
        "--min-segs", type=int, default=MIN_SEGMENTS,
        help=f"Min segments to consider a cluster (default: {MIN_SEGMENTS})",
    )
    parser.add_argument(
        "--speakers", nargs="+", metavar="ID",
        help="Review only these speakers (e.g. SPEAKER_04 SPEAKER_13)",
    )
    parser.add_argument(
        "--audio-track", type=int, default=None,
        help="Audio track index (0-based within audio streams) for DUAL releases",
    )
    parser.add_argument(
        "--apply", action="store_true",
        help="Write results to label files",
    )

    args = parser.parse_args()

    # Dispatch based on --clips vs Vision
    if args.clips:
        def process_fn(ep, vid):
            return review_episode_interactive(
                    ep, vid, args.min_segs, args.apply, args.audio_track, args.speakers,
                )
    else:
        def process_fn(ep, vid):
            return process_episode_vision(
                    ep, vid, args.samples, args.min_segs, args.apply,
                )

    if args.episode:
        if not args.video:
            parser.error("--video is required with --episode")
        if not Path(args.video).exists():
            parser.error(f"Video not found: {args.video}")
        process_fn(args.episode, args.video)

    elif args.series and args.season is not None:
        if not args.video_dir:
            parser.error("--video-dir is required for batch mode")
        video_dir = Path(args.video_dir)
        if not video_dir.is_dir():
            parser.error(f"Not a directory: {args.video_dir}")

        episodes = discover_episodes(args.series, args.season)
        if not episodes:
            print(f"No diarization data for {args.series.upper()} season {args.season}")
            sys.exit(1)

        print(f"Found {len(episodes)} episodes")

        for ep_id in episodes:
            video = find_video(video_dir, ep_id)
            if video:
                process_fn(ep_id, str(video))
            else:
                print(f"\n  {ep_id}: no video found in {video_dir}, skipping")

    else:
        parser.error(
            "Provide --episode + --video (single) or "
            "--series + --season + --video-dir (batch)"
        )


if __name__ == "__main__":
    main()
