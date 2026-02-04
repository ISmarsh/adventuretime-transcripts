# PGS Subtitle OCR Workflow

This document describes the process for extracting text from Blu-ray PGS (bitmap) subtitles.

## Key Finding

**Blu-ray PGS subtitles include speaker labels** (FINN:, JAKE:, BMO:, etc.) for some lines. This provides partial speaker attribution that the wiki-scraped transcripts lack.

## Requirements

- **ffmpeg** - Extract PGS subtitle stream from MKV
- **Tesseract OCR** - `winget install UB-Mannheim.TesseractOCR`
- **Python 3.10+** with PIL/Pillow and pytesseract

## Process

### 1. Extract PGS Stream

```bash
ffmpeg -i "video.mkv" -map 0:s:0 -c:s copy "output.sup"
```

### 2. OCR the SUP File

Use the `pgs_to_srt.py` script to convert SUP to SRT:

```bash
python tools/pgs_to_srt.py input.sup output.srt
```

### 3. Manual Cleanup Required

OCR output typically has errors:
- Music notes (♪) misrecognized as letters (J, ~, f)
- Character substitutions: "h" → "n", "i" → "l"
- Stylized fonts cause garbled text
- Approximately 10-20% of lines need correction

## OCR Results Summary

All 10 episodes needing speaker attribution were processed. Islands miniseries (S08) yielded useful speaker labels; Elements miniseries (S09) had minimal labels. Overall ~60-70% of OCR lines are clean, ~30-40% need manual correction.

## Comparison: OCR vs Wiki-Scraped

| Source | Speaker Labels | Text Quality | Effort |
|--------|----------------|--------------|--------|
| Wiki-scraped | Minimal (1-10 labels) | Good | Done |
| PGS OCR | Partial (10-20 labels) | Mixed | High cleanup |
| Manual video | Full | Perfect | Very high |

## Recommendation

PGS OCR is useful for:
1. Extracting speaker labels not in wiki transcripts
2. Cross-referencing dialogue text
3. Getting timestamps for speaker attribution

Not recommended as sole source due to OCR errors. Best used to supplement wiki transcripts.

## Files

- `tools/pgs_to_srt.py` - Python script for PGS to SRT conversion
- Requires: `pip install pillow pytesseract`

## Related Documentation

- [TRANSCRIPT-CORRECTIONS.md](TRANSCRIPT-CORRECTIONS.md) - Tracks correction work needed
- [TRANSCRIPT-GAPS.md](TRANSCRIPT-GAPS.md) - Documents transcript sources
