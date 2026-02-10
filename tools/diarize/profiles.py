"""Voice profile I/O: load, save, index management, season filtering."""

import json
from datetime import datetime
from pathlib import Path

from .config import BUCKET_RE, ECAPA_SOURCE, EMBED_DIM


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


def _load_profile_sample_counts(profile_dir: Path) -> dict[str, int]:
    """Load sample counts for all profiles from the index file."""
    index_path = profile_dir / "_index.json"
    if not index_path.exists():
        return {}
    index = json.loads(index_path.read_text(encoding="utf-8"))
    return {
        name: info.get("samples", 0)
        for name, info in index.get("profiles", {}).items()
    }


def _load_profile_first_episodes(profile_dir: Path) -> dict[str, str]:
    """Load first_episode constraints from the index file.

    Returns a dict mapping profile name -> earliest episode ID where the
    character exists (e.g. "AT.S06E01").  Profiles without the field are
    omitted — they match any episode.
    """
    index_path = profile_dir / "_index.json"
    if not index_path.exists():
        return {}
    index = json.loads(index_path.read_text(encoding="utf-8"))
    return {
        name: info["first_episode"]
        for name, info in index.get("profiles", {}).items()
        if "first_episode" in info
    }


def _load_profile_last_episodes(profile_dir: Path) -> dict[str, str]:
    """Load last_episode constraints from the index file.

    Returns a dict mapping profile name -> latest episode ID where the
    profile is valid (e.g. "AT.S10E13").  Profiles without the field are
    omitted — they match any episode.
    """
    index_path = profile_dir / "_index.json"
    if not index_path.exists():
        return {}
    index = json.loads(index_path.read_text(encoding="utf-8"))
    return {
        name: info["last_episode"]
        for name, info in index.get("profiles", {}).items()
        if "last_episode" in info
    }


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


def _rebuild_index(profile_dir: Path) -> None:
    """Scan all .npz profiles and rebuild _index.json, preserving custom fields."""
    index_path = profile_dir / "_index.json"
    existing_index: dict = {}
    if index_path.exists():
        existing_index = json.loads(
            index_path.read_text(encoding="utf-8")
        ).get("profiles", {})

    profiles_meta: dict = {}
    for npz_path in profile_dir.glob("*.npz"):
        prof = _load_profile(npz_path)
        episodes_in = list({
            m.get("episode", "") for m in prof["metadata"]
        })
        entry = dict(existing_index.get(npz_path.stem, {}))
        entry["samples"] = len(prof["embeddings"])
        entry["episodes"] = sorted(e for e in episodes_in if e)
        entry["updated"] = datetime.now().isoformat(timespec="seconds")
        profiles_meta[npz_path.stem] = entry

    _save_index(profile_dir, profiles_meta)
    return len(profiles_meta)


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
