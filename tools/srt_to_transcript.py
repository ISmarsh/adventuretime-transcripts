#!/usr/bin/env python3
"""Convert SRT subtitle files to plain transcript format.

Strips SRT index numbers and timing lines, merges multi-line entries,
preserves any speaker labels (BOTH:, FINN:, etc.), and collapses
consecutive blank lines.
"""

import argparse
import re
import sys
from pathlib import Path

# Matches SRT index lines (just a number on its own line)
INDEX_RE = re.compile(r'^\d+$')

# Matches SRT timing lines: 00:00:00,000 --> 00:00:00,000
TIMING_RE = re.compile(r'^\d{2}:\d{2}:\d{2},\d{3}\s*-->\s*\d{2}:\d{2}:\d{2},\d{3}')

# Matches SDH speaker labels like "FINN:", "BOTH:", "PRINCESS BUBBLEGUM:"
SPEAKER_RE = re.compile(r'^([A-Z][A-Z\s\'.]+):(.+)')

# Common SDH sound/music descriptions to strip: [music], (door slams), ♪ lyrics ♪
SOUND_RE = re.compile(r'^[\[\(♪].*[\]\)♪]$')


def parse_srt(text: str) -> list[str]:
    """Parse SRT content into a list of dialogue lines.

    Each SRT entry may span multiple text lines. These are joined with
    a space. Index and timing lines are stripped entirely.
    """
    lines = text.splitlines()
    entries: list[str] = []
    current: list[str] = []

    for line in lines:
        stripped = line.strip()

        # Skip index and timing lines
        if INDEX_RE.match(stripped):
            # Flush any accumulated text
            if current:
                entries.append(' '.join(current))
                current = []
            continue

        if TIMING_RE.match(stripped):
            continue

        if stripped == '':
            # Blank line = entry separator
            if current:
                entries.append(' '.join(current))
                current = []
            continue

        current.append(stripped)

    # Flush remaining
    if current:
        entries.append(' '.join(current))

    return entries


def is_sound_description(line: str) -> bool:
    """Check if a line is a pure sound/music description (not dialogue)."""
    return bool(SOUND_RE.match(line.strip()))


def format_speaker_label(line: str) -> str:
    """Convert SDH speaker label to transcript format.

    SDH uses "SPEAKER: dialogue" (uppercase, one space).
    Transcript format uses "Speaker:  dialogue" (title case, two spaces).
    """
    m = SPEAKER_RE.match(line)
    if m:
        speaker = m.group(1).strip().title()
        dialogue = m.group(2).strip()
        return f"{speaker}:  {dialogue}"
    return line


def convert_srt_to_transcript(srt_text: str, *, keep_sounds: bool = False) -> str:
    """Convert SRT content to plain transcript format."""
    entries = parse_srt(srt_text)

    output: list[str] = []
    for entry in entries:
        # Optionally filter pure sound descriptions
        if not keep_sounds and is_sound_description(entry):
            continue

        # Format any speaker labels
        formatted = format_speaker_label(entry)
        output.append(formatted)

    return '\n'.join(output) + '\n'


def main():
    parser = argparse.ArgumentParser(
        description="Convert SRT subtitle files to plain transcript format.",
    )
    parser.add_argument('input', help="Input SRT file")
    parser.add_argument('output', nargs='?', help="Output transcript file (default: stdout)")
    parser.add_argument(
        '--keep-sounds', action='store_true',
        help="Keep sound/music descriptions (default: strip them)",
    )
    args = parser.parse_args()

    srt_path = Path(args.input)
    srt_text = srt_path.read_text(encoding='utf-8')

    result = convert_srt_to_transcript(srt_text, keep_sounds=args.keep_sounds)

    if args.output:
        out_path = Path(args.output)
        out_path.write_text(result, encoding='utf-8')
        print(f"Wrote {len(result.splitlines())} lines to {out_path}")
    else:
        sys.stdout.write(result)


if __name__ == '__main__':
    main()
