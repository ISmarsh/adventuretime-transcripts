"""Filesystem scanning, episode discovery, and utility functions."""

import logging
import shutil
import sys
from difflib import SequenceMatcher
from pathlib import Path

from .config import (
    _NORM_RE,
    _SPACE_RE,
    EPISODE_RE,
    FC_PATTERN,
    MULTI_EP_RE,
    TRANSCRIPT_SERIES_DIRS,
    Episode,
)

logger = logging.getLogger("whisperx_diarize")


# ---------------------------------------------------------------------------
# Utility functions
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
