# Adventure Time Transcripts

Transcripts for Adventure Time and spin-off series, scraped from the [Adventure Time Wiki](https://adventuretime.fandom.com).

## Structure

```
Adventure Time/
  Season 00 (Specials)/  - 2 files
  Season 01-10/          - 258 files

Adventure Time Distant Lands/
  Season 01/             - 4 files

Adventure Time Fionna and Cake/
  Season 01/             - 10 files
  Season 02/             - 10 files
```

Total: 282 transcript files

## Gap Filling

Some episodes were missing or incomplete on the wiki. These gaps were filled by converting SRT/SDH subtitle files. See [TRANSCRIPT-GAPS.md](TRANSCRIPT-GAPS.md) for details.

## Documentation

- [TRANSCRIPT-GAPS.md](TRANSCRIPT-GAPS.md) - Which episodes were filled from subtitles
- [TRANSCRIPT-CORRECTIONS.md](TRANSCRIPT-CORRECTIONS.md) - Corrections and attribution work needed
- [VERIFICATION-WORKFLOW.md](VERIFICATION-WORKFLOW.md) - Process for verifying speaker attribution
- [PGS-OCR-WORKFLOW.md](PGS-OCR-WORKFLOW.md) - Extracting text from Blu-ray bitmap subtitles

## Tools

- `tools/pgs_to_srt.py` - Convert PGS (Blu-ray bitmap) subtitles to SRT using OCR

## Format

Wiki transcript format:
- Two spaces after speaker colon: `Character:  dialogue`
- Scene descriptions in brackets: `[Scene description.]`
- Inline actions: `Character:  [gasps] Dialogue here.`
