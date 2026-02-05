#!/usr/bin/env python3
"""Validate transcript files against the gold-standard format.

Usage:
    # Validate a single file:
    python tools/validate_transcript.py transcripts/Adventure\ Time/Season\ 01/Adventure.Time.S01E01.Slumber.Party.Panic.txt

    # Validate all files in a directory:
    python tools/validate_transcript.py --batch transcripts/

    # Cross-reference speakers against characters.json:
    python tools/validate_transcript.py --characters src/data/characters.json FILE
"""

import argparse
import json
import re
import sys
from pathlib import Path


# Speaker line pattern (must match parse_transcript_speakers.py)
SPEAKER_PATTERN = re.compile(r"^([A-Z][^:\[\]]+?):\s", re.MULTILINE)

# Two-space rule: speaker label must have exactly two spaces after colon
TWO_SPACE_PATTERN = re.compile(r"^([A-Z][^:\[\]]+?):(\s+)")

# SRT artifacts
SRT_TIMING_PATTERN = re.compile(r"^\d{2}:\d{2}:\d{2}[,\.]\d{3}")
SRT_INDEX_PATTERN = re.compile(r"^\d+$")

# Wiki markup remnants
WIKI_PATTERNS = [
    (re.compile(r"'''"), "bold markup (''')"),
    (re.compile(r"''(?!')"), "italic markup ('')"),
    (re.compile(r"\[\[[^\]]*\]\]"), "wiki link ([[...]])"),
    (re.compile(r"\{\{[^}]*\}\}"), "template ({{...}})"),
    (re.compile(r"^=+.*=+$"), "section header (==...==)"),
]

# HTML remnants
HTML_PATTERN = re.compile(r"</?[a-z][^>]*>", re.IGNORECASE)


def validate_file(filepath: Path, characters: dict | None = None) -> list[dict]:
    """Validate a single transcript file.

    Returns list of issues, each with:
        line: line number (1-indexed)
        type: issue category
        message: human-readable description
        severity: 'error' | 'warning'
    """
    issues = []
    text = filepath.read_text(encoding="utf-8")
    lines = text.split("\n")

    prev_blank = False
    has_speaker_lines = False
    unmatched_speakers = set()

    for i, line in enumerate(lines, 1):
        # Check for double-spacing (SRT artifact)
        if not line.strip():
            if prev_blank and i < len(lines):
                issues.append({
                    "line": i,
                    "type": "srt_artifact",
                    "message": "Multiple consecutive blank lines (SRT double-spacing artifact)",
                    "severity": "warning",
                })
            prev_blank = True
            continue
        prev_blank = False

        # Check for SRT timing lines
        if SRT_TIMING_PATTERN.match(line):
            issues.append({
                "line": i,
                "type": "srt_artifact",
                "message": f"SRT timing line: {line[:30]}",
                "severity": "error",
            })
            continue

        # Check for SRT index numbers (standalone digit lines)
        stripped = line.strip()
        if SRT_INDEX_PATTERN.match(stripped) and len(stripped) <= 4:
            # Could be a legitimate line with just a number, but flag it
            issues.append({
                "line": i,
                "type": "srt_artifact",
                "message": f"Possible SRT index number: {stripped}",
                "severity": "warning",
            })

        # Check speaker label formatting
        two_space_match = TWO_SPACE_PATTERN.match(line)
        if two_space_match:
            has_speaker_lines = True
            spaces = two_space_match.group(2)
            if len(spaces) != 2:
                issues.append({
                    "line": i,
                    "type": "spacing",
                    "message": f"Speaker label has {len(spaces)} space(s) after colon, expected 2: {line[:50]}",
                    "severity": "error",
                })

            # Check speaker against characters
            speaker = two_space_match.group(1).strip()
            if characters and speaker.lower() not in characters:
                unmatched_speakers.add(speaker)

        # Check for wiki markup remnants
        for pattern, name in WIKI_PATTERNS:
            if pattern.search(line):
                issues.append({
                    "line": i,
                    "type": "wiki_markup",
                    "message": f"Wiki markup remnant ({name}): {line[:60]}",
                    "severity": "error",
                })

        # Check for HTML remnants
        if HTML_PATTERN.search(line):
            issues.append({
                "line": i,
                "type": "html",
                "message": f"HTML tag found: {line[:60]}",
                "severity": "error",
            })

        # Check for unattributed lines (??? markers)
        if line.startswith("???:"):
            issues.append({
                "line": i,
                "type": "unattributed",
                "message": f"Unattributed speaker: {line[:60]}",
                "severity": "error",
            })

        # Check for bare dialogue (line that's not a speaker label, not a stage
        # direction, and not blank — may indicate missing speaker attribution)
        if (
            not SPEAKER_PATTERN.match(line)
            and not line.strip().startswith("[")
            and not line.strip().startswith("(")
            and len(line.strip()) > 2
        ):
            # Don't flag lines that are clearly continuation or stage directions
            # This is a heuristic — many transcripts have stage directions
            # without brackets in wiki format
            pass

    # No speaker lines at all = raw SRT
    if not has_speaker_lines and len(lines) > 10:
        issues.append({
            "line": 0,
            "type": "format",
            "message": "No speaker labels found — file appears to be raw SRT without attribution",
            "severity": "error",
        })

    # Report unmatched speakers
    if unmatched_speakers:
        for speaker in sorted(unmatched_speakers):
            issues.append({
                "line": 0,
                "type": "unknown_speaker",
                "message": f"Speaker not in characters.json: {speaker}",
                "severity": "warning",
            })

    return issues


def load_character_names(characters_file: Path) -> dict:
    """Load character names/aliases for cross-referencing."""
    with open(characters_file, encoding="utf-8") as f:
        characters = json.load(f)

    name_map = {}
    for char in characters:
        if char.get("type") == "group":
            continue
        name_map[char["name"].lower()] = char["id"]
        for alias in char.get("aliases", []):
            name_map[alias.lower()] = char["id"]

    return name_map


def main():
    parser = argparse.ArgumentParser(description="Validate transcript format")
    parser.add_argument(
        "files",
        nargs="*",
        type=Path,
        help="Transcript file(s) to validate",
    )
    parser.add_argument(
        "--batch",
        type=Path,
        help="Validate all .txt files in directory recursively",
    )
    parser.add_argument(
        "--characters",
        type=Path,
        help="Path to characters.json for speaker cross-referencing",
    )
    parser.add_argument(
        "--errors-only",
        action="store_true",
        help="Only show errors, not warnings",
    )

    args = parser.parse_args()

    # Load characters if provided
    characters = None
    if args.characters:
        print(f"Loading characters from {args.characters}...")
        characters = load_character_names(args.characters)
        print(f"  {len(characters)} name/alias mappings loaded")

    # Collect files to validate
    files = list(args.files) if args.files else []
    if args.batch:
        files.extend(sorted(args.batch.rglob("*.txt")))

    if not files:
        parser.error("No files specified. Use positional args or --batch")

    # Validate each file
    total_errors = 0
    total_warnings = 0
    files_with_issues = 0

    for filepath in files:
        if not filepath.exists():
            print(f"\nERROR: File not found: {filepath}")
            total_errors += 1
            continue

        issues = validate_file(filepath, characters)

        if args.errors_only:
            issues = [i for i in issues if i["severity"] == "error"]

        errors = [i for i in issues if i["severity"] == "error"]
        warnings = [i for i in issues if i["severity"] == "warning"]

        if issues:
            files_with_issues += 1
            print(f"\n{filepath.name}:")
            for issue in issues:
                prefix = "ERROR" if issue["severity"] == "error" else "WARN"
                line_str = f"L{issue['line']}" if issue["line"] > 0 else "   "
                print(f"  [{prefix}] {line_str}: {issue['message']}")

            total_errors += len(errors)
            total_warnings += len(warnings)

    # Summary
    print(f"\n{'=' * 60}")
    print(f"Validated {len(files)} file(s)")
    print(f"  {files_with_issues} file(s) with issues")
    print(f"  {total_errors} error(s), {total_warnings} warning(s)")

    if total_errors > 0:
        print("\nFAILED — errors found")
        sys.exit(1)
    elif total_warnings > 0:
        print("\nPASSED with warnings")
    else:
        print("\nPASSED — all clean")


if __name__ == "__main__":
    main()
