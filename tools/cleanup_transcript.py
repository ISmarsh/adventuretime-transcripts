#!/usr/bin/env python3
"""Normalize transcript formatting for consistency across all files.

Pure str → str transforms. Run this LAST, after all content changes
(speaker attribution, line merging) are complete.

Transforms applied:
    1. Normalize line endings → LF
    2. Normalize brackets — [ text ] → [text]
    3. Normalize dashes — ‐‐ (U+2010×2) → --, leading —/‐ → -
    4. Fix speaker colon spacing — ensure exactly two spaces after colon
    5. Separate joined bracket lines — ][ → ]\\n[
    6. Merge same-speaker subtitle fragments — join when continuation starts lowercase
    7. Warn on joined OCR words — detect [a-z][A-Z], print to stderr
    8. Collapse blank lines — max 1 consecutive blank line
    9. Trim trailing whitespace, ensure final newline

Usage:
    python cleanup_transcript.py <file_or_dir...>           # preview diff
    python cleanup_transcript.py <file_or_dir...> --write   # write in-place
    python cleanup_transcript.py <file_or_dir...> --stats   # show change counts
"""

import argparse
import io
import re
import sys
from pathlib import Path

# Force UTF-8 output on Windows
if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if sys.stderr.encoding != "utf-8":
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# Speaker label: Name(s) followed by colon at start of line
SPEAKER_RE = re.compile(r"^([A-Z][A-Za-z0-9 \u2019'.\-()]+):\s")

# Joined OCR words: lowercase immediately followed by uppercase (e.g., "helloWorld")
JOINED_WORD_RE = re.compile(r"[a-z][A-Z]")


def normalize(text: str) -> str:
    """Apply all formatting transforms to transcript text."""
    # 1. LF line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    lines = text.split("\n")
    result: list[str] = []

    for line in lines:
        # 8a. Trim trailing whitespace
        line = line.rstrip()

        # 2. Normalize brackets: [ text ] → [text], but not [  ] (empty)
        line = re.sub(r"\[\s+", "[", line)
        line = re.sub(r"\s+\]", "]", line)

        # 3. Normalize dashes
        line = line.replace("\u2010\u2010", "--")  # ‐‐ → --
        # Leading em-dash or hyphen-minus to "- " for stage direction prefix
        line = re.sub(r"^[\u2014\u2013\u2010]\s*", "- ", line)

        # 4. Fix speaker colon spacing — exactly two spaces after speaker: label
        m = SPEAKER_RE.match(line)
        if m:
            speaker = m.group(1).rstrip()
            rest = line[m.end():].lstrip()
            line = f"{speaker}:  {rest}"

        # 5. Separate joined bracket lines: ][ → ]\n[
        if "][" in line:
            parts = line.split("][")
            expanded = ("]\n[".join(parts))
            result.extend(expanded.split("\n"))
            continue

        result.append(line)

    # 6. Merge consecutive same-speaker subtitle fragments.
    # Only merge when the continuation line's dialogue starts with a lowercase letter,
    # which indicates a mid-sentence break from subtitle line limits.
    merged: list[str] = []
    i = 0
    while i < len(result):
        line = result[i]
        m = SPEAKER_RE.match(line)
        if not m:
            merged.append(line)
            i += 1
            continue

        speaker = m.group(1).rstrip()
        dialogue = line[m.end():].lstrip()

        # Try to absorb following same-speaker continuations
        while True:
            j = i + 1
            # Skip exactly one blank line
            if j < len(result) and result[j].strip() == "":
                j += 1
            if j >= len(result):
                break
            m2 = SPEAKER_RE.match(result[j])
            if not m2:
                break
            speaker2 = m2.group(1).rstrip()
            if speaker != speaker2:
                break
            dialogue2 = result[j][m2.end():].lstrip()
            if not dialogue2 or not dialogue2[0].islower():
                break
            # Merge: append continuation text
            dialogue = dialogue + " " + dialogue2
            i = j  # skip blank line + merged line

        merged.append(f"{speaker}:  {dialogue}")
        i += 1

    result = merged

    # 8. Collapse consecutive blank lines (max 1)
    collapsed: list[str] = []
    prev_blank = False
    for line in result:
        is_blank = line.strip() == ""
        if is_blank and prev_blank:
            continue
        collapsed.append(line)
        prev_blank = is_blank

    # 9b. Strip leading/trailing blank lines, ensure final newline
    while collapsed and collapsed[0].strip() == "":
        collapsed.pop(0)
    while collapsed and collapsed[-1].strip() == "":
        collapsed.pop()

    return "\n".join(collapsed) + "\n"


def warn_joined_words(path: Path, text: str) -> int:
    """Check for joined OCR words and print warnings. Returns count."""
    count = 0
    for i, line in enumerate(text.split("\n"), 1):
        stripped = line.strip()
        # Skip scene descriptions and music lines
        if stripped.startswith("[") or stripped.startswith("\u266a"):
            continue
        matches = JOINED_WORD_RE.findall(stripped)
        if matches:
            # Filter out common false positives
            false_positives = {"BMO", "McCoy", "McDonald", "McFly", "iPhone", "iPad"}
            real_matches = []
            for m in matches:
                # Check if this is inside a known word
                is_fp = False
                for fp in false_positives:
                    if m in fp and fp in stripped:
                        is_fp = True
                        break
                if not is_fp:
                    real_matches.append(m)
            if real_matches:
                count += 1
                print(
                    f"  {path.name}:{i}: possible joined words: {stripped[:80]}",
                    file=sys.stderr,
                )
    return count


def collect_files(paths: list[str]) -> list[Path]:
    """Expand arguments into list of .txt transcript files."""
    files: list[Path] = []
    for p in paths:
        path = Path(p)
        if path.is_dir():
            files.extend(sorted(path.rglob("*.txt")))
        elif path.is_file() and path.suffix == ".txt":
            files.append(path)
        else:
            print(f"Warning: skipping {path}", file=sys.stderr)
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", help="Transcript files or directories")
    parser.add_argument("--write", action="store_true", help="Write changes in-place")
    parser.add_argument("--stats", action="store_true", help="Show change counts only")
    args = parser.parse_args()

    files = collect_files(args.paths)
    if not files:
        print("No .txt files found.", file=sys.stderr)
        sys.exit(1)

    total_changed = 0
    total_joined = 0

    for path in files:
        original = path.read_text(encoding="utf-8", errors="replace")
        cleaned = normalize(original)

        changed = original != cleaned
        if changed:
            total_changed += 1

        # Warn on joined OCR words
        joined = warn_joined_words(path, cleaned)
        total_joined += joined

        if args.stats:
            if changed:
                # Count line-level diffs
                orig_lines = original.split("\n")
                clean_lines = cleaned.split("\n")
                diff_count = sum(
                    1 for a, b in zip(orig_lines, clean_lines) if a != b
                ) + abs(len(orig_lines) - len(clean_lines))
                print(f"  {path.name}: {diff_count} lines changed")
            continue

        if changed and not args.write:
            # Preview mode: show diff
            print(f"\n--- {path}")
            orig_lines = original.split("\n")
            clean_lines = cleaned.split("\n")
            shown = 0
            for j, (a, b) in enumerate(zip(orig_lines, clean_lines), 1):
                if a != b and shown < 10:
                    print(f"  L{j}:")
                    print(f"    - {a[:100]}")
                    print(f"    + {b[:100]}")
                    shown += 1
            if shown == 10:
                remaining = sum(
                    1 for a, b in zip(orig_lines, clean_lines) if a != b
                ) - 10
                if remaining > 0:
                    print(f"  ... and {remaining} more changes")

        if changed and args.write:
            path.write_text(cleaned, encoding="utf-8")
            print(f"  Updated: {path.name}")

    print(f"\nSummary: {total_changed}/{len(files)} files changed", file=sys.stderr)
    if total_joined:
        print(f"  {total_joined} possible joined-word warnings", file=sys.stderr)
    if not args.write and total_changed:
        print("  Use --write to apply changes.", file=sys.stderr)


if __name__ == "__main__":
    main()
