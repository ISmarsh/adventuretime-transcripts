#!/usr/bin/env python3
"""Fetch Adventure Time transcripts from the Fandom wiki API and format to standard.

Usage:
    # Attempt API fetch:
    python tools/wiki_to_transcript.py --api "Crossover" --season 7 --episode 23

    # Paste mode (pipe wikitext from clipboard or file):
    python tools/wiki_to_transcript.py --paste --season 7 --episode 23 --title "Crossover" < pasted.txt

    # Process a raw wikitext file:
    python tools/wiki_to_transcript.py --file raw.txt --season 7 --episode 23 --title "Crossover"

Output is saved to the correct season directory with the standard filename convention.
"""

import argparse
import re
import sys
from pathlib import Path

# Only import requests if available (not required for paste/file modes)
try:
    import requests

    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


FANDOM_API = "https://adventuretime.fandom.com/api.php"

# Series prefixes for filename construction
SERIES_PREFIXES = {
    "at": "Adventure.Time",
    "dl": "Adventure.Time.Distant.Lands",
    "fc": "Adventure.Time.Fionna.and.Cake",
}


def fetch_wiki_transcript(episode_title: str) -> str | None:
    """Attempt to fetch transcript wikitext from the Fandom API.

    Returns wikitext string on success, None on failure.
    """
    if not HAS_REQUESTS:
        print("  requests library not installed, skipping API fetch")
        return None

    # Try common page name patterns
    page_patterns = [
        f"{episode_title}/Transcript",
        f"{episode_title.replace(' ', '_')}/Transcript",
    ]

    for page_name in page_patterns:
        url = FANDOM_API
        params = {
            "action": "parse",
            "page": page_name,
            "format": "json",
            "prop": "wikitext",
        }

        try:
            resp = requests.get(url, params=params, timeout=15)
            if resp.status_code == 200:
                data = resp.json()
                if "parse" in data and "wikitext" in data["parse"]:
                    wikitext = data["parse"]["wikitext"]["*"]
                    if wikitext and len(wikitext.strip()) > 50:
                        print(f"  Fetched {len(wikitext)} chars from wiki API")
                        return wikitext
            elif resp.status_code == 403:
                print(f"  API returned 403 Forbidden for: {page_name}")
                return None
            else:
                print(f"  API returned {resp.status_code} for: {page_name}")
        except requests.RequestException as e:
            print(f"  API request failed: {e}")
            return None

    return None


def strip_wiki_markup(text: str) -> str:
    """Remove wiki markup from a text fragment.

    Handles: links, bold/italic, HTML tags, entities.
    """
    # Convert wiki links: [[Page|Display]] -> Display, [[Page]] -> Page
    text = re.sub(r"\[\[([^|\]]*)\|([^\]]*)\]\]", r"\2", text)
    text = re.sub(r"\[\[([^\]]*)\]\]", r"\1", text)

    # Remove bold/italic wiki markup
    text = text.replace("'''", "")
    text = text.replace("''", "")

    # Remove HTML tags
    text = re.sub(r"<br\s*/?>", " ", text)
    text = re.sub(r"</?[a-z][^>]*>", "", text, flags=re.IGNORECASE)

    # HTML entities
    text = text.replace("&mdash;", "—")
    text = text.replace("&ndash;", "–")
    text = text.replace("&amp;", "&")
    text = text.replace("&lt;", "<")
    text = text.replace("&gt;", ">")
    text = text.replace("&#39;", "'")
    text = text.replace("&quot;", '"')

    return text.strip()


def parse_l_template(line: str) -> str | None:
    """Parse a {{L|...}} dialogue template line.

    Formats:
        {{L|Speaker|Dialogue text}}           -> Speaker:  Dialogue text
        {{L|[Stage direction]}}               -> [Stage direction]
        {{L|Speaker|[action] Dialogue}}       -> Speaker:  [action] Dialogue
        {{L|Speaker and Speaker2|Dialogue}}   -> Speaker and Speaker2:  Dialogue
    """
    line = line.strip()
    if not line.startswith("{{L|"):
        return None

    # Remove outer {{L| and closing }}
    inner = line[4:]
    if inner.endswith("}}"):
        inner = inner[:-2]

    # Split on first pipe to get speaker vs dialogue
    # But be careful: inner content may have nested templates or pipes in links
    # Use a simple approach: find the first | that's not inside [[ ]]
    depth_bracket = 0
    depth_brace = 0
    split_pos = None

    for i, ch in enumerate(inner):
        if ch == "[":
            depth_bracket += 1
        elif ch == "]":
            depth_bracket -= 1
        elif ch == "{":
            depth_brace += 1
        elif ch == "}":
            depth_brace -= 1
        elif ch == "|" and depth_bracket == 0 and depth_brace == 0:
            split_pos = i
            break

    if split_pos is None:
        # No pipe — this is a stage direction: {{L|[description]}}
        content = strip_wiki_markup(inner)
        # Ensure stage directions are in brackets
        if content and not content.startswith("["):
            content = f"[{content}]"
        return content

    speaker_raw = inner[:split_pos]
    dialogue_raw = inner[split_pos + 1:]

    speaker = strip_wiki_markup(speaker_raw).strip()
    dialogue = strip_wiki_markup(dialogue_raw).strip()

    if not speaker:
        return dialogue if dialogue else None

    if dialogue:
        return f"{speaker}:  {dialogue}"
    else:
        return f"{speaker}:"


def clean_wikitext(text: str) -> str:
    """Convert wikitext to clean transcript format.

    Handles two wiki formats:
    1. {{L|Speaker|Dialogue}} template format (most transcript pages)
    2. Raw Speaker: Dialogue format (some older pages)

    Also strips: section headers, metadata templates, wiki markup, HTML.
    """
    lines = text.split("\n")
    cleaned = []
    in_transcript = False

    for line in lines:
        line = line.rstrip()

        # Skip section headers but note when we enter the transcript section
        if re.match(r"^=+\s*Transcript\s*=+$", line, re.IGNORECASE):
            in_transcript = True
            continue
        if re.match(r"^=+.*=+$", line):
            # Another section header after transcript = end of transcript
            if in_transcript:
                break
            continue

        # Skip the metadata template block ({{Transcript ...}})
        if line.strip().startswith("{{Transcript"):
            # Skip until closing }}
            if "}}" not in line:
                # Multi-line template — skip until we find }}
                continue
            continue
        # Skip lines that are part of a multi-line metadata template
        if not in_transcript and line.strip().startswith("|"):
            continue
        if line.strip() == "}}":
            continue

        # Handle {{L|...}} template lines
        l_result = parse_l_template(line)
        if l_result is not None:
            in_transcript = True
            cleaned.append(l_result)
            continue

        # Handle raw format lines (Speaker: Dialogue)
        if in_transcript:
            # Strip remaining wiki markup from raw lines
            line = strip_wiki_markup(line)

            # Normalize speaker labels: ensure exactly two spaces after colon
            speaker_match = re.match(r"^([A-Z][^:\[\]]{0,50}):(\s*)(.*)", line)
            if speaker_match:
                speaker = speaker_match.group(1).strip()
                dialogue = speaker_match.group(3).strip()
                if dialogue:
                    line = f"{speaker}:  {dialogue}"
                else:
                    line = f"{speaker}:"

            line = line.rstrip()

            # Skip completely empty lines in sequence
            if not line and cleaned and not cleaned[-1]:
                continue

            cleaned.append(line)

    # Strip leading/trailing blank lines
    while cleaned and not cleaned[0]:
        cleaned.pop(0)
    while cleaned and not cleaned[-1]:
        cleaned.pop()

    return "\n".join(cleaned) + "\n"


def build_filename(
    season: int, episode: int, title: str, series: str = "at"
) -> str:
    """Build standard transcript filename.

    Convention: Adventure.Time.SxxExx.Title.Here.txt
    - Dots separate words
    - Preserve & ' ! + , -
    - Lowercase articles mid-title (the, of, a)
    """
    prefix = SERIES_PREFIXES.get(series, "Adventure.Time")

    # Clean title for filename
    words = title.split()
    filename_words = []
    for i, word in enumerate(words):
        # Lowercase mid-title articles
        if i > 0 and word.lower() in ("the", "of", "a", "an", "in", "on", "to"):
            filename_words.append(word.lower())
        else:
            filename_words.append(word)

    title_part = ".".join(filename_words)
    # Remove any characters that shouldn't be in filenames
    title_part = re.sub(r'[?:"<>|]', "", title_part)

    return f"{prefix}.S{season:02d}E{episode:02d}.{title_part}.txt"


def get_output_dir(season: int, series: str = "at") -> Path:
    """Get the output directory for a transcript file."""
    tools_dir = Path(__file__).parent
    transcripts_dir = tools_dir.parent

    series_dirs = {
        "at": "Adventure Time",
        "dl": "Adventure Time Distant Lands",
        "fc": "Adventure Time Fionna and Cake",
    }
    series_dir = series_dirs.get(series, "Adventure Time")
    return transcripts_dir / series_dir / f"Season {season:02d}"


def main():
    parser = argparse.ArgumentParser(
        description="Fetch and format Adventure Time wiki transcripts"
    )

    # Input mode
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--api", metavar="TITLE", help="Fetch from Fandom API by episode title"
    )
    group.add_argument(
        "--paste", action="store_true", help="Read wikitext from stdin"
    )
    group.add_argument(
        "--file", type=Path, help="Read wikitext from a file"
    )

    # Episode metadata
    parser.add_argument("--season", type=int, required=True, help="Season number")
    parser.add_argument("--episode", type=int, required=True, help="Episode number")
    parser.add_argument("--title", help="Episode title (required for paste/file modes)")
    parser.add_argument(
        "--series",
        choices=["at", "dl", "fc"],
        default="at",
        help="Series: at=Adventure Time, dl=Distant Lands, fc=Fionna and Cake",
    )

    # Output
    parser.add_argument(
        "--output", type=Path, help="Custom output path (default: auto-detect)"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print to stdout instead of saving"
    )

    args = parser.parse_args()

    # Determine episode title
    episode_title = args.api or args.title
    if not episode_title:
        parser.error("--title is required for paste/file modes")

    # Get wikitext
    wikitext = None

    if args.api:
        print(f"Fetching transcript for '{episode_title}'...")
        wikitext = fetch_wiki_transcript(episode_title)
        if wikitext is None:
            print("\nAPI fetch failed. Options:")
            print(f"  1. Open in browser: https://adventuretime.fandom.com/wiki/{episode_title.replace(' ', '_')}/Transcript")
            print("  2. Copy the transcript text")
            print(f"  3. Run: python tools/wiki_to_transcript.py --paste --season {args.season} --episode {args.episode} --title \"{episode_title}\"")
            sys.exit(1)
    elif args.paste:
        print("Reading wikitext from stdin (paste content, then Ctrl+D / Ctrl+Z)...")
        wikitext = sys.stdin.read()
    elif args.file:
        print(f"Reading wikitext from {args.file}...")
        wikitext = args.file.read_text(encoding="utf-8")

    if not wikitext or not wikitext.strip():
        print("Error: No content to process")
        sys.exit(1)

    # Clean and format
    print("Formatting transcript...")
    transcript = clean_wikitext(wikitext)

    # Count speaker lines for summary
    speaker_lines = len(re.findall(r"^[A-Z][^:\[\]]+?:\s", transcript, re.MULTILINE))
    total_lines = len([line for line in transcript.split("\n") if line.strip()])
    print(f"  {total_lines} content lines, {speaker_lines} with speaker labels")

    if args.dry_run:
        print("\n--- OUTPUT ---")
        print(transcript)
        return

    # Determine output path
    filename = build_filename(args.season, args.episode, episode_title, args.series)
    if args.output:
        output_path = args.output
    else:
        output_dir = get_output_dir(args.season, args.series)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / filename

    # Save
    output_path.write_text(transcript, encoding="utf-8")
    print(f"\nSaved: {output_path}")
    print(f"Filename: {filename}")


if __name__ == "__main__":
    main()
