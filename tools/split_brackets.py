#!/usr/bin/env python3
"""Extract cross-speaker audible bracketed actions to standalone lines.

Finds lines like: `Princess Bubblegum:  You up to this, champ? [Finn coughs] Ok, well...`
Splits to:
    Princess Bubblegum:  You up to this, champ?
    [Finn coughs]
    Princess Bubblegum:  Ok, well...

Only targets audible actions (laughs, gasps, screams, etc.) where the
bracketed character is a known speaker in the file and differs from
the line's speaker.
"""

import io
import re
import sys
from pathlib import Path

# Fix Windows encoding
if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(
        sys.stdout.buffer, encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRANSCRIPT_DIRS = [
    PROJECT_ROOT / "Adventure Time",
    PROJECT_ROOT / "Adventure Time Distant Lands",
    PROJECT_ROOT / "Adventure Time Fionna and Cake",
]

SPEAKER_RE = re.compile(r"^([A-Z][A-Za-z0-9 \u2019'.\-()]+):\s\s")

AUDIBLE_VERBS = (
    r"laughs|laughing|gasps|gasping|screams|screaming|sighs|sighing|"
    r"cries|crying|grunts|grunting|groans|groaning|shrieks|shrieking|"
    r"coughs|coughing|snores|snoring|giggles|giggling|chuckles|chuckling|"
    r"sobs|sobbing|whimpers|whimpering|yells|yelling|shouts|shouting|"
    r"hums|humming|sings|singing|whistles|whistling|sneezes|sneezing|"
    r"burps|burping|hiccups|moans|moaning|growls|growling|whines|whining|"
    r"wails|wailing|yelps|squeals|squealing"
)

# Match: [Name (optional modifier) audible_verb ...]
BRACKET_RE = re.compile(
    r'\[([A-Z][a-zA-Z\'-]+)'                                # Name (capitalized word)
    r'\s+'
    r'(?:(?:still|starts?|begins?|continues?|keeps?)\s+)?'   # optional modifier
    r'(?:' + AUDIBLE_VERBS + r')'                            # audible verb
    r'[^\]]*\]'                                              # rest of bracket
)


def names_match(bracket_name: str, line_speaker: str) -> bool:
    """Check if the bracket name refers to the same speaker."""
    bn = bracket_name.lower()
    ls = line_speaker.lower()
    if bn == ls:
        return True
    if bn in ls.split():
        return True
    return False


def collect_speaker_words(lines: list[str]) -> set[str]:
    """Collect all speaker names and their component words from a file."""
    names = set()
    for line in lines:
        m = SPEAKER_RE.match(line)
        if m:
            full = m.group(1)
            names.add(full)
            for word in full.split():
                if len(word) >= 3:  # skip "of", "St", etc.
                    names.add(word)
    return names


def process_line(line: str, known_speakers: set[str]):
    """Process one line. Returns (output_lines, changes).

    Splits dialogue around cross-speaker brackets to preserve temporal order:
        Finn:  text1 [Jake laughs] text2
    becomes:
        Finn:  text1
        [Jake laughs]
        Finn:  text2
    """
    m = SPEAKER_RE.match(line)
    if not m:
        return [line], []

    speaker = m.group(1)
    prefix = f"{speaker}:  "
    content_start = m.end()

    # Find cross-speaker audible brackets
    to_extract = []
    for bm in BRACKET_RE.finditer(line):
        bracket_name = bm.group(1)
        if bracket_name not in known_speakers:
            continue
        if names_match(bracket_name, speaker):
            continue
        to_extract.append(bm)

    if not to_extract:
        return [line], []

    # Split line at each extracted bracket, preserving order
    result = []
    pos = content_start

    for bm in to_extract:
        # Dialogue segment before this bracket
        before = line[pos:bm.start()].strip()
        if before:
            result.append(prefix + before)
        # The bracket itself as a standalone line
        result.append(bm.group(0))
        pos = bm.end()

    # Dialogue segment after the last bracket
    after = line[pos:].strip()
    if after:
        result.append(prefix + after)

    changes = [{
        "speaker": speaker,
        "original": line,
        "extracted": [bm.group(0) for bm in to_extract],
        "result_lines": result,
    }]
    return result, changes


def process_file(filepath: Path):
    """Process a transcript file. Returns changes made."""
    text = filepath.read_text(encoding="utf-8")
    lines = text.split("\n")

    # First pass: collect known speaker names
    known_speakers = collect_speaker_words(lines)

    # Second pass: process lines
    new_lines = []
    all_changes = []

    for i, line in enumerate(lines):
        result, changes = process_line(line, known_speakers)
        new_lines.extend(result)
        for c in changes:
            c["file"] = str(filepath)
            c["line_num"] = i + 1
            all_changes.append(c)

    if all_changes:
        result_text = "\n".join(new_lines)
        if text.endswith("\n") and not result_text.endswith("\n"):
            result_text += "\n"
        filepath.write_text(result_text, encoding="utf-8")

    return all_changes


def main():
    total = []
    for tdir in TRANSCRIPT_DIRS:
        if not tdir.exists():
            continue
        for txt in sorted(tdir.rglob("*.txt")):
            changes = process_file(txt)
            total.extend(changes)

    print(f"Total lines split: {len(total)}")
    for c in total:
        rel = Path(c["file"]).relative_to(PROJECT_ROOT)
        print(f"\n{rel}:{c['line_num']}")
        print(f"  ORIGINAL: {c['original'][:140]}")
        print("  BECOMES:")
        for rl in c["result_lines"]:
            print(f"    {rl[:140]}")


if __name__ == "__main__":
    main()
