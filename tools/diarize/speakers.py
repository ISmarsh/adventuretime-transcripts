"""Speaker name resolution: alias maps, canonical names, profile bucketing."""

import json
import logging

from .config import (
    BUCKET_RE,
    CANONICAL_SPEAKER,
    ENCHIRIDION_CHARACTERS,
    MANUAL_ALIASES,
    SEASON_BUCKETS,
    VOICE_SEPARATE,
)

logger = logging.getLogger("whisperx_diarize")

_alias_cache: dict[str, str] | None = None


def _load_alias_map() -> dict[str, str]:
    """Build speaker alias -> canonical name map from characters.json."""
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


def _canon(name: str) -> str:
    """Canonicalize a speaker name for validation comparison.

    Strips season-bucket suffix (Finn_S01-S03 -> Finn) and series-bucket
    suffix (Prismo_FC -> Prismo) then resolves transcript aliases
    (Lumpy Space Princess -> LSP).
    """
    m = BUCKET_RE.match(name)
    if m:
        name = m.group(1)
    # Strip series-bucket suffix (e.g. Prismo_FC, Gary_FC)
    for suffix in ("_FC", "_DL"):
        if name.endswith(suffix):
            name = name[:-len(suffix)]
            break
    return _resolve_speaker(name)


def _get_profile_name(character: str, season: int) -> str:
    """Return bucketed profile name (e.g. 'Finn_S01-S03') or plain name."""
    if character in SEASON_BUCKETS:
        for s, e in SEASON_BUCKETS[character]:
            if s <= season <= e:
                return f"{character}_S{s:02d}-S{e:02d}"
    return character
