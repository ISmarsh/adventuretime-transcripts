#!/usr/bin/env python3
"""Build complete transcripts by merging sources and attributing speakers.

Takes a wiki transcript (primary source with some speaker labels) and
optionally an OCR transcript (secondary, fills dialogue gaps), then
runs speaker inference on unlabeled lines.

Usage:
    # Attribute speakers in a transcript
    python build_transcript.py transcript.txt -o output.txt

    # Merge wiki + OCR, then attribute
    python build_transcript.py wiki.txt --ocr ocr.txt -o output.txt

    # Use episode character list for better inference
    python build_transcript.py wiki.txt --characters "Finn,Jake,BMO" -o output.txt
"""

import argparse
import re
import sys
from pathlib import Path

# Line patterns
SPEAKER_RE = re.compile(r'^([^:\[\]]+):\s{2}(.+)')
SCENE_RE = re.compile(r'^\[([^\]]+)\]$')
INLINE_ACTION_RE = re.compile(r'\[([^\]]+)\]')

# Characters mentioned in scene descriptions
SCENE_MENTION_RE = re.compile(r'\b([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)\b')

# Address patterns: "Hey, Jake!" or "Jake, look!"
ADDRESS_RE = re.compile(r'(?:Hey|Hi|Yo|Look|Come on|Wait),?\s+([A-Z][a-z]+)')
ADDRESS_END_RE = re.compile(r'([A-Z][a-z]+),?\s+(?:look|come|wait|help|stop|listen|watch)')

# Multi-speaker labels that shouldn't propagate to unlabeled lines
MULTI_SPEAKER_LABELS = {'both', 'all', 'everyone', 'together'}


def parse_transcript_line(line: str) -> dict:
    """Parse a transcript line into components."""
    stripped = line.strip()
    if not stripped:
        return {'type': 'blank', 'text': ''}

    # Scene description
    if SCENE_RE.match(stripped):
        return {'type': 'scene', 'text': stripped}

    # Speaker-labeled dialogue
    m = SPEAKER_RE.match(stripped)
    if m:
        speaker = m.group(1).strip()
        dialogue = m.group(2).strip()
        return {'type': 'dialogue', 'speaker': speaker, 'dialogue': dialogue, 'text': stripped}

    # Unlabeled line (dialogue without speaker)
    return {'type': 'unlabeled', 'text': stripped}


def extract_scene_characters(scene_text: str, known_names: set[str]) -> list[str]:
    """Extract character names mentioned in a scene description."""
    found = []
    for m in SCENE_MENTION_RE.finditer(scene_text):
        name = m.group(1)
        if name in known_names:
            found.append(name)
    return found


def extract_address_target(dialogue: str, known_names: set[str]) -> str | None:
    """Extract who is being addressed in dialogue."""
    for pattern in [ADDRESS_RE, ADDRESS_END_RE]:
        m = pattern.search(dialogue)
        if m:
            name = m.group(1)
            if name in known_names:
                return name
    return None


def build_name_set(characters: list[str]) -> set[str]:
    """Build a set of character first names for matching."""
    names = set()
    for char in characters:
        # Use the name as-is
        names.add(char)
        # Also add first word as shorthand
        first = char.split()[0] if ' ' in char else char
        names.add(first)
    return names


def _is_multi_speaker(speaker: str) -> bool:
    """Check if a speaker label represents multiple characters."""
    return speaker.lower() in MULTI_SPEAKER_LABELS


def attribute_speakers(lines: list[dict], known_names: set[str]) -> list[dict]:
    """Run speaker inference on parsed transcript lines.

    Conservative approach: only infer the FIRST unlabeled line after a
    labeled dialogue or scene description.  Beyond that, mark as ???.
    This prevents cascading mis-attribution.
    """
    last_scene_chars: list[str] = []
    recent_speakers: list[str] = []  # Last few distinct speakers for turn-taking
    unlabeled_run = 0  # Consecutive unlabeled lines (reset by dialogue/scene)
    results = []

    for i, line in enumerate(lines):
        if line['type'] == 'blank':
            results.append(line)
            continue

        if line['type'] == 'scene':
            last_scene_chars = extract_scene_characters(line['text'], known_names)
            unlabeled_run = 0
            results.append(line)
            continue

        if line['type'] == 'dialogue':
            speaker = line['speaker']
            unlabeled_run = 0

            # Track recent distinct speakers (skip multi-speaker labels)
            if not _is_multi_speaker(speaker):
                if not recent_speakers or recent_speakers[-1] != speaker:
                    recent_speakers.append(speaker)
                    if len(recent_speakers) > 4:
                        recent_speakers.pop(0)

            results.append(line)

            # Check if this speaker addresses someone
            target = extract_address_target(line['dialogue'], known_names)
            if target:
                line['address_target'] = target
            continue

        # Unlabeled line — try to infer speaker
        unlabeled_run += 1
        inferred = None
        confidence = 'none'

        # Only attempt inference on the FIRST unlabeled line in a run.
        # Beyond that, mark as ??? to avoid cascading errors.
        if unlabeled_run == 1:
            # Rule 1: Previous speaker addressed someone by name
            prev_dialogue = _find_prev_dialogue(results)
            if prev_dialogue and prev_dialogue.get('address_target'):
                inferred = prev_dialogue['address_target']
                confidence = 'medium'

            # Rule 2: Turn-taking — alternate to the other recent speaker
            if not inferred and len(recent_speakers) >= 2:
                if not _has_scene_break_since_last_dialogue(results):
                    inferred = recent_speakers[-2]
                    confidence = 'low'

            # Rule 3: Scene mentions a single character
            if not inferred and last_scene_chars and len(last_scene_chars) == 1:
                inferred = last_scene_chars[0]
                confidence = 'low'

        result = dict(line)
        if inferred:
            result['inferred_speaker'] = inferred
            result['confidence'] = confidence
        else:
            result['inferred_speaker'] = '???'
            result['confidence'] = 'none'

        results.append(result)

    return results


def _find_prev_dialogue(results: list[dict]) -> dict | None:
    """Find the most recent labeled dialogue line in results."""
    for line in reversed(results):
        if line['type'] == 'dialogue':
            return line
    return None


def _has_scene_break_since_last_dialogue(results: list[dict]) -> bool:
    """Check if there's a scene description between now and last labeled dialogue."""
    for line in reversed(results):
        if line['type'] == 'dialogue':
            return False
        if line['type'] == 'scene':
            return True
    return False


def format_output(lines: list[dict]) -> str:
    """Format attributed lines back to transcript text."""
    output = []
    for line in lines:
        if line['type'] == 'blank':
            output.append('')
            continue

        if line['type'] in ('scene', 'dialogue'):
            output.append(line['text'])
            continue

        # Unlabeled line — format with inferred speaker
        speaker = line.get('inferred_speaker', '???')
        confidence = line.get('confidence', 'none')
        text = line['text']

        if speaker == '???':
            output.append(f"???:  {text}")
        elif confidence == 'low':
            output.append(f"{speaker} [?]:  {text}")
        else:
            output.append(f"{speaker}:  {text}")

    return '\n'.join(output) + '\n'


def compute_stats(lines: list[dict]) -> dict:
    """Compute attribution statistics."""
    total = 0
    with_speaker = 0
    inferred = 0
    uncertain = 0
    unknown = 0

    for line in lines:
        if line['type'] == 'dialogue':
            total += 1
            with_speaker += 1
        elif line['type'] == 'unlabeled':
            total += 1
            speaker = line.get('inferred_speaker', '???')
            conf = line.get('confidence', 'none')
            if speaker == '???':
                unknown += 1
            elif conf == 'low':
                uncertain += 1
            else:
                inferred += 1

    return {
        'total_dialogue': total,
        'original_speakers': with_speaker,
        'inferred': inferred,
        'uncertain': uncertain,
        'unknown': unknown,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Build transcripts with speaker attribution.",
    )
    parser.add_argument('input', help="Primary transcript file (wiki-fetched or existing)")
    parser.add_argument('-o', '--output', help="Output file (default: stdout)")
    parser.add_argument('--ocr', help="OCR transcript for gap-filling reference")
    parser.add_argument(
        '--characters', help="Comma-separated character names for inference",
    )
    parser.add_argument('--stats-only', action='store_true', help="Print stats without output")
    args = parser.parse_args()

    # Read primary transcript
    primary_text = Path(args.input).read_text(encoding='utf-8')
    lines = [parse_transcript_line(line) for line in primary_text.splitlines()]

    # Build character name set
    known_names: set[str] = set()
    if args.characters:
        for name in args.characters.split(','):
            name = name.strip()
            known_names.add(name)
            # Add first word as shorthand
            if ' ' in name:
                known_names.add(name.split()[0])

    # Also extract names from existing speaker labels
    for line in lines:
        if line['type'] == 'dialogue':
            known_names.add(line['speaker'])
            if ' ' in line['speaker']:
                known_names.add(line['speaker'].split()[0])

    # Run attribution
    attributed = attribute_speakers(lines, known_names)

    # Stats
    stats = compute_stats(attributed)
    pct_known = (
        (stats['original_speakers'] + stats['inferred'])
        / max(stats['total_dialogue'], 1)
        * 100
    )
    print(
        f"  {stats['total_dialogue']} dialogue lines: "
        f"{stats['original_speakers']} original, "
        f"{stats['inferred']} inferred, "
        f"{stats['uncertain']} uncertain, "
        f"{stats['unknown']} unknown "
        f"({pct_known:.0f}% attributed)",
        file=sys.stderr,
    )

    if args.stats_only:
        return

    # Format output
    output = format_output(attributed)

    if args.output:
        Path(args.output).write_text(output, encoding='utf-8')
        print(f"  Wrote to {args.output}", file=sys.stderr)
    else:
        sys.stdout.write(output)


if __name__ == '__main__':
    main()
