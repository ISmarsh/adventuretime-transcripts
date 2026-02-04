# Transcript Corrections

Tracks corrections made to previously existing transcripts.

## Format Standardization

- Removed `==Transcript==` headers from 17 files
- Renamed 9 files to remove special characters (`?`, `:`)

## Speaker Attribution Needed

The following episodes have raw dialogue without speaker labels:

| Episode | Lines | Notes |
|---------|-------|-------|
| S08E21 Whipple the Happy Dragon | 456 | Islands miniseries |
| S08E23 Imaginary Resources | 414 | Islands miniseries |
| S08E25 Min and Marty | 583 | Islands miniseries |
| S08E27 The Light Cloud | 505 | Islands miniseries |
| S09E01 Orb | 548 | |
| S09E04 Winter Light | 441 | Elements miniseries |
| S09E05 Cloudy | 522 | Elements miniseries |
| S09E06 Slime Central | 435 | Elements miniseries |
| S09E09 Skyhooks II | 395 | Elements miniseries |
| S09E11 Ketchup | 433 | |

## Verification Methods

- **Video frame extraction** - Extract frames at timestamp to verify speaker
- **SDH subtitles** - Use speaker labels from SDH tracks when available
- **PGS OCR** - Extract text from Blu-ray bitmap subtitles (see [PGS-OCR-WORKFLOW.md](PGS-OCR-WORKFLOW.md))
- **Context clues** - Scene setup, dialogue references, character patterns

## Notes

Main series Blu-ray has PGS (bitmap) subtitles which include some speaker labels. See [PGS-OCR-WORKFLOW.md](PGS-OCR-WORKFLOW.md) for extraction process.

**Subtitle sources:**
- Blu-ray PGS (bitmap, needs OCR)
- SUBDL (subdl.com) - may have text-based subtitles

## Tools

- `tools/pgs_to_srt.py` - Python script for PGS to SRT conversion with OCR
