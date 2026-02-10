"""Spot-check subcommand: interactive review of validation disagrees via VLC."""

import argparse
import io
import json
import re
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

from .config import (
    DEFAULT_VIDEO_DIRS,
    EPISODE_RE,
    PROGRESS_FILE,
    PROJECT_ROOT,
    SPEAKER_RE,
)
from .cmd_validate import (
    discover_for_validate,
    load_progress,
    validate_episode,
)
from .discovery import _best_score, _normalize, scan_videos
from .speakers import _canon
from .vlc import extract_clip, find_vlc, play_clips_vlc, stop_vlc

SPOT_CHECK_DIR = "spot_check"


def _clipboard(text: str) -> None:
    """Copy text to system clipboard (Windows)."""
    if sys.platform == "win32":
        subprocess.run(
            ["clip"], input=text.encode(),
            creationflags=subprocess.CREATE_NO_WINDOW,
        )


def _resolve_video_path(ep: dict, video_index: dict) -> str | None:
    """Resolve video path for an episode via diarization JSON then video index."""
    # From diarization JSON
    dia_json = ep.get("dia_json")
    if dia_json and dia_json.exists():
        data = json.loads(dia_json.read_text(encoding="utf-8"))
        vp = data.get("video_path", "")
        if vp and Path(vp).exists():
            return vp

    # From video index (scan_videos)
    eid = ep["episode_id"]
    m = EPISODE_RE.search(eid.replace(".", ""))
    if m:
        code = eid.split(".")[0] if "." in eid else "AT"
        key = (code, int(m.group(1)), int(m.group(2)))
        if key in video_index:
            return str(video_index[key])

    return None


def _load_spot_check(dia_dir: Path, episode_id: str) -> dict:
    """Load existing spot-check decisions for resume."""
    path = dia_dir / SPOT_CHECK_DIR / f"{episode_id}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"episode_id": episode_id, "decisions": {}}


def _save_spot_check(dia_dir: Path, data: dict) -> None:
    """Save spot-check decisions (resume-safe)."""
    sc_dir = dia_dir / SPOT_CHECK_DIR
    sc_dir.mkdir(parents=True, exist_ok=True)
    path = sc_dir / f"{data['episode_id']}.json"
    data["updated"] = datetime.now().isoformat(timespec="seconds")
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _apply_corrections(transcript_path: Path, corrections: dict[int, str]) -> int:
    """Apply speaker corrections to a transcript file.

    corrections: {line_num: correct_speaker_name}
    Returns count of lines changed.
    """
    original = transcript_path.read_text(encoding="utf-8")
    lines = original.split("\n")
    changed = 0

    for line_num, correct_speaker in corrections.items():
        idx = line_num - 1  # line_num is 1-based
        if idx < 0 or idx >= len(lines):
            continue
        raw = lines[idx]
        m = SPEAKER_RE.match(raw)
        if m:
            # Preserve the two-space format: "Speaker:  dialogue"
            dialogue_start = m.end()
            lines[idx] = f"{correct_speaker}:  {raw[dialogue_start:]}"
            changed += 1

    text = "\n".join(lines)
    if not text.endswith("\n"):
        text += "\n"
    transcript_path.write_text(text, encoding="utf-8")
    return changed


def spot_check_episode(
    ep: dict, video_path: str, dia_dir: Path,
    label_dir: Path | None = None,
    audio_track: int | None = None,
    limit: int | None = None,
    force: bool = False,
) -> dict:
    """Interactive spot-check of disagrees for one episode. Returns decisions."""
    eid = ep["episode_id"]
    vlc_path = find_vlc()
    if not vlc_path:
        print("Error: VLC not found. Install VLC or add it to PATH.",
              file=sys.stderr)
        sys.exit(1)

    # Re-run validation to get TLine objects with timestamps
    result, lines, original = validate_episode(ep, label_dir)

    if result.error:
        print(f"  Validation error: {result.error}")
        return {}

    # Build disagree list with timestamps
    disagrees = []
    for tl in lines:
        if tl.validation == "disagree" and tl.w_start >= 0:
            # Find the cluster->character mapping from result
            dia_speaker = ""
            conf = 0.0
            for d in result.disagree_details:
                if d["line"] == tl.line_num:
                    dia_speaker = d["diarization"]
                    conf = d["conf"]
                    break
            disagrees.append({
                "tline": tl,
                "diarization_speaker": dia_speaker,
                "conf": conf,
            })

    if not disagrees:
        print(f"  No disagrees with timestamps for {eid}")
        return {}

    # Load existing decisions for resume
    sc_data = _load_spot_check(dia_dir, eid)
    existing = sc_data.get("decisions", {})

    # Filter out already-reviewed (unless --force)
    if not force:
        remaining = [
            d for d in disagrees if str(d["tline"].line_num) not in existing
        ]
    else:
        remaining = disagrees

    if limit:
        remaining = remaining[:limit]

    print(f"\n{'=' * 60}")
    print(f"Episode: {eid} -- {ep.get('title', '')}")
    print(f"Video:   {video_path}")
    print(f"Disagrees: {len(disagrees)} total, {len(existing)} reviewed, "
          f"{len(remaining)} remaining")
    print(f"\nCommands: transcript / diarization / name <Speaker> / "
          f"skip / replay / quit (or first letter)")
    print()

    # Build line index for context display
    line_index = {tl.line_num: tl for tl in lines}

    decisions = dict(existing)
    vlc_proc = None

    with tempfile.TemporaryDirectory() as tmpdir:
        for i, item in enumerate(remaining, 1):
            tl = item["tline"]
            dia_speaker = item["diarization_speaker"]
            conf = item["conf"]

            # Find best-matching whisperX segment text in time range
            wx_text = ""
            tl_norm = _normalize(tl.text)
            wx_best_r = 0.0
            for seg in ep.get("segments", []):
                if seg["start"] <= tl.w_end and seg["end"] >= tl.w_start:
                    stxt = seg.get("text", "").strip()
                    sr = _best_score(tl_norm, _normalize(stxt))
                    if sr > wx_best_r:
                        wx_best_r = sr
                        wx_text = stxt
                elif seg["start"] > tl.w_end:
                    break

            # Show context lines (2 before, 2 after)
            print(f"--- [{i}/{len(remaining)}] Line {tl.line_num} ---")
            for offset in range(-2, 3):
                ctx = line_index.get(tl.line_num + offset)
                if not ctx or ctx.is_scene:
                    continue
                if offset == 0:
                    marker = ">>"
                else:
                    marker = "  "
                print(f"  {marker} L{ctx.line_num} {ctx.speaker}: "
                      f"{ctx.text[:60]}")

            if wx_text:
                print(f'  WhisperX:   "{wx_text[:80]}"')
            print(f"  Transcript says:   {tl.speaker}")
            print(f"  Diarization says:  {dia_speaker} "
                  f"(cluster {tl.dia_speaker}, conf {conf:.0%})")
            print(f"  Time: {tl.w_start:.1f}s - {tl.w_end:.1f}s")

            # Extract and play clip
            clip_path = str(Path(tmpdir) / f"line_{tl.line_num}.mp4")
            if extract_clip(video_path, tl.w_start, tl.w_end, clip_path,
                            audio_track):
                stop_vlc(vlc_proc)
                vlc_proc = play_clips_vlc(vlc_path, [clip_path])
            else:
                print("  (clip extraction failed)")

            custom_speaker = None
            while True:
                answer = input(
                    "\n  Decision (transcript / diarization / name <Speaker>"
                    " / skip / replay / quit): "
                ).strip()

                answer_lower = answer.lower()
                if answer_lower in ("replay", "r"):
                    stop_vlc(vlc_proc)
                    if Path(clip_path).exists():
                        vlc_proc = play_clips_vlc(vlc_path, [clip_path])
                    continue
                elif answer_lower.startswith("name ") or \
                        answer_lower.startswith("n "):
                    # Custom speaker name (preserve case from input)
                    parts = answer.split(None, 1)
                    if len(parts) == 2 and parts[1].strip():
                        custom_speaker = parts[1].strip()
                        stop_vlc(vlc_proc)
                        vlc_proc = None
                        answer = "name"
                        break
                    else:
                        print("  Usage: name <Speaker Name>")
                        continue
                elif answer_lower in ("transcript", "t", "diarization", "d",
                                      "skip", "s", "quit", "q"):
                    answer = answer_lower
                    stop_vlc(vlc_proc)
                    vlc_proc = None
                    break
                else:
                    print("  Invalid. Use: transcript (t), diarization (d), "
                          "name <Speaker> (n), skip (s), replay (r), "
                          "quit (q)")

            if answer in ("quit", "q"):
                print("  Quitting.")
                break

            # Normalize to full word
            decision_map = {
                "t": "transcript", "transcript": "transcript",
                "d": "diarization", "diarization": "diarization",
                "s": "skip", "skip": "skip",
                "name": "name",
            }
            decision = decision_map[answer]

            entry = {
                "decision": decision,
                "transcript_speaker": tl.speaker,
                "diarization_speaker": dia_speaker,
                "text": tl.text[:80],
                "cluster": tl.dia_speaker,
                "ts": datetime.now().isoformat(timespec="seconds"),
            }
            if custom_speaker:
                entry["custom_speaker"] = custom_speaker
            decisions[str(tl.line_num)] = entry

            label = {
                "transcript": f"-> TRANSCRIPT correct ({tl.speaker})",
                "diarization": f"-> DIARIZATION correct ({dia_speaker})",
                "name": f"-> CUSTOM: {custom_speaker}",
                "skip": "-> skipped",
            }
            print(f"  {label[decision]}")

            # Save after each decision (resume-safe)
            sc_data["decisions"] = decisions
            sc_data["title"] = ep.get("title", "")
            _save_spot_check(dia_dir, sc_data)

    # Summary
    counts = {"transcript": 0, "diarization": 0, "name": 0, "skip": 0}
    for d in decisions.values():
        dec = d.get("decision", "skip") if isinstance(d, dict) else d
        if dec in counts:
            counts[dec] += 1

    summary = (
        f"Summary for {eid}:\n"
        f"  Transcript correct: {counts['transcript']}\n"
        f"  Diarization correct: {counts['diarization']}\n"
        + (f"  Custom name: {counts['name']}\n"
           if counts['name'] else "")
        + f"  Skipped: {counts['skip']}\n"
        f"  Total reviewed: {sum(counts.values())}/{len(disagrees)}"
    )
    print(f"\n{summary}")
    _clipboard(summary)
    print("  (copied to clipboard)")

    return decisions


def cmd_spot_check(args: argparse.Namespace) -> None:
    """Interactive review of validation disagrees via VLC clips."""
    if sys.stdout.encoding != "utf-8":
        sys.stdout = io.TextIOWrapper(
            sys.stdout.buffer, encoding="utf-8", errors="replace")
    if sys.stderr.encoding != "utf-8":
        sys.stderr = io.TextIOWrapper(
            sys.stderr.buffer, encoding="utf-8", errors="replace")

    dia_dir = PROJECT_ROOT / args.diarization_dir
    if not dia_dir.exists():
        print(f"Error: {dia_dir} not found", file=sys.stderr)
        sys.exit(1)

    # Use validation progress to find episodes with disagrees
    progress_path = dia_dir / PROGRESS_FILE
    progress = load_progress(progress_path)

    # Discover episodes
    episodes = discover_for_validate(PROJECT_ROOT, dia_dir)

    # Apply filters
    if args.episode:
        episodes = [e for e in episodes if e["episode_id"] in args.episode]
    if args.series:
        prefix = args.series.upper() + "."
        episodes = [e for e in episodes if e["episode_id"].startswith(prefix)]
    if args.season is not None:
        episodes = [e for e in episodes if e["season"] == args.season]

    # Filter to episodes with known disagrees
    if not args.force:
        filtered = []
        for ep in episodes:
            eid = ep["episode_id"]
            ep_prog = progress.get("processed", {}).get(eid, {})
            if ep_prog.get("disagreements"):
                filtered.append(ep)
        episodes = filtered

    if args.limit:
        episodes = episodes[:args.limit]

    if not episodes:
        print("No episodes with disagrees to spot-check.")
        print("Run 'validate' first to identify disagreements.")
        return

    # Resolve video paths
    video_dirs = getattr(args, "video_dirs", DEFAULT_VIDEO_DIRS)
    video_index = scan_videos(video_dirs)

    label_dir = dia_dir / "labels"
    if not label_dir.exists():
        label_dir = None

    audio_track = getattr(args, "audio_track", None)

    print(f"Found {len(episodes)} episodes with disagrees to review\n")

    for i, ep in enumerate(episodes, 1):
        eid = ep["episode_id"]

        video_path = _resolve_video_path(ep, video_index)
        if not video_path:
            print(f"[{i}/{len(episodes)}] {eid} -- SKIP (no video found)")
            continue

        decisions = spot_check_episode(
            ep, video_path, dia_dir, label_dir,
            audio_track=audio_track,
            force=args.force,
        )

        # Apply corrections if --write
        if args.write and decisions:
            corrections = {}
            for line_str, d in decisions.items():
                if not isinstance(d, dict):
                    continue
                if d.get("decision") == "diarization":
                    # Clean bucket name before writing to transcript
                    speaker = _canon(d["diarization_speaker"])
                    corrections[int(line_str)] = speaker
                elif d.get("decision") == "name" and d.get("custom_speaker"):
                    corrections[int(line_str)] = d["custom_speaker"]

            if corrections:
                changed = _apply_corrections(ep["transcript"], corrections)
                print(f"  WROTE {changed} corrections to "
                      f"{ep['transcript'].name}")

    print(f"\n{'=' * 60}")
    print("Spot-check complete.")
