# Transcript Corrections

Tracks corrections made to previously existing transcripts.

## Format Standardization

- Removed `==Transcript==` headers from 17 files
- Renamed 9 files to remove special characters (`?`, `:`)

## Speaker Attribution — Completed

Speaker labels were added to these episodes via automated Vision AI attribution
(Claude Haiku 4.5 + video frame extraction). Coverage ~84% across all 10 episodes;
remaining unlabeled lines are mostly songs, sound effects, or ambiguous dialogue.

| Episode | Dialogue | Labeled | Coverage | Notes |
|---------|----------|---------|----------|-------|
| S08E21 Whipple the Happy Dragon | 200 | 144 | 72.0% | Islands miniseries |
| S08E23 Imaginary Resources | 170 | 130 | 76.5% | Islands miniseries |
| S08E25 Min and Marty | 210 | 166 | 79.0% | Islands miniseries |
| S08E27 The Light Cloud | 207 | 160 | 77.3% | Islands miniseries |
| S09E01 Orb | 206 | 175 | 85.0% | |
| S09E04 Winter Light | 194 | 181 | 93.3% | Elements miniseries |
| S09E05 Cloudy | 242 | 220 | 90.9% | Elements miniseries |
| S09E06 Slime Central | 184 | 167 | 90.8% | Elements miniseries |
| S09E09 Skyhooks II | 178 | 151 | 84.8% | Elements miniseries |
| S09E11 Ketchup | 201 | 178 | 88.6% | |
| **Total** | **1992** | **1672** | **83.9%** | |

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

## Review Pass — Post-Attribution Fixes

After Vision attribution and format cleanup, a review pass identified and fixed:

### Vision-leaked speaker descriptions

The Vision AI sometimes appended character descriptions to speaker names. These were stripped:

| File | Original Label | Fixed Label | Lines |
|------|---------------|-------------|-------|
| F&C S01E07 The Star | `Chef (bakery owner in apron):` | `Chef:` | 6 |
| F&C S01E07 The Star | `Bonnie (pink-haired character):` | `Bonnie:` | 3 |
| F&C S01E07 The Star | `Pink-haired character (Bonnie):` | `Bonnie:` | 1 |
| F&C S01E07 The Star | `Old man (elderly character in tuxedo):` | `Old Man:` | 1 |
| F&C S01E01 Fionna Campbell | `Bartender (Sammy):` | `Norm:` | 1 |
| S09E05 Cloudy | `Finn (character in blue shirt):` | `Finn:` | 1 |
| S08E27 The Light Cloud | `Kara (Minerva bot):` | `Kara:` | 1 |
| S09E11 Ketchup | `Character (appears to be one of the people inside the tranch):` | `Person:` | 1 |

### PGS OCR all-caps speaker names

S08E25 (Min and Marty) had all speaker labels in ALL CAPS from PGS subtitle OCR. Fixed to title case:
`MINERVA` (53), `MAN` (26), `DR. GROSS` (18), `SEEKER` (5), `TOGETHER` (2), `KARA` (1), `FINN` (1), `MARTIN` (1), `MARTY` (2).

Also split a mid-line speaker change: `MINERVA: Oh! -- MARTIN: You're all right, Doc.`

**Not fixed (intentional):** S08E23 all-caps MMO usernames (`WOLFPRUD3`, `R3M3MB3RCATS`, `THEINFAMO`, `TOOTER9`, `MOTHERSCHILD`) — these are in-universe screen names.

### BMO casing

S08E23: Fixed 4 instances of `Bmo:` to `BMO:` (OCR miscasing).

### Known remaining issues

- **OCR joined words (`[a-z][A-Z]`)**: All 14 instances fixed manually (`saidI`→`said I`, `bespokeIce`→`bespoke Ice`, `MarcelineThanks`→`Marceline:  Thanks`, etc.). Zero `[a-z][A-Z]` join warnings remain.
- **OCR joined words (`[a-z][a-z]`)**: 164 lowercase-only joins fixed via dictionary-based splitting across 10 PGS-OCR'd episodes (S08E21–S09E11). Used 370K English word list + transcript corpus vocabulary to identify words not in any dictionary that split into two known words. Heaviest: S09E05 Cloudy (58 auto + 17 manual = 75 fixes). Remaining 3 flagged items are intentional non-fixes (`thali`, `Iassi`, `gallbag`).
- **Same-speaker line fragmentation**: Merged via lowercase-start heuristic in `cleanup_transcript.py`. 38/282 files affected (20 attributed + 18 wiki files with numbered speaker names like `Lemongrab 2`).
- **S08E25 `Man:` label**: Generic PGS label for multiple male characters — most are Martin Mertens but some may be other characters.

### Dictionary-based OCR join splitting

PGS OCR frequently omits spaces between words, producing joins invisible to the `[a-z][A-Z]` detector. A dictionary-based approach was used:

1. Load 370K English word list as primary "is this a real word?" filter
2. Build supplementary vocabulary from all 282 transcript files (AT-specific terms)
3. For each word in OCR files: if NOT in dictionary/corpus → try splitting at all positions
4. Both halves must be known words; score by corpus frequency with function-word bonus

| Episode | Auto (HIGH) | Manual | Total |
|---------|-------------|--------|-------|
| S08E21 Whipple the Happy Dragon | 2 | 0 | 2 |
| S08E23 Imaginary Resources | 27 | 7 | 34 |
| S08E27 The Light Cloud | 7 | 0 | 7 |
| S09E03 Bespoken For | 3 | 0 | 3 |
| S09E04 Winter Light | 16 | 5 | 21 |
| S09E05 Cloudy | 58 | 17 | 75 |
| S09E06 Slime Central | 9 | 6 | 15 |
| S09E07 Happy Warrior | 1 | 0 | 1 |
| S09E09 Skyhooks II | 4 | 1 | 5 |
| S09E11 Ketchup | 1 | 0 | 1 |
| **Total** | **128** | **36** | **164** |

Manual fixes included: wrong-split corrections (`formulatea`→`formulate a` not `formula tea`), single-char right halves (`witha`→`with a`, `justa`→`just a`, `likea`→`like a`), and contraction joins (`Doesn'tmake`→`Doesn't make`, `there'sthat`→`there's that`).

**Not fixed (intentional):** `thali` (Indian food term), `Iassi` (OCR for lassi), `gallbag` (AT compound).

## Format Cleanup

All 282 transcript files were normalized with `tools/cleanup_transcript.py --write`:
- Bracket spacing: `[ text ]` → `[text]`
- Speaker colon spacing: exactly two spaces after colon (`Speaker:  dialogue`)
- Unicode dash normalization: `‐‐` → `--`
- Joined bracket separation: `][` → `]\n[`
- Same-speaker subtitle merge: consecutive same-speaker lines merged when continuation starts lowercase
- Collapsed consecutive blank lines
- LF line endings, final newline

Second pass: 38/282 files updated (speaker merge + digit-name colon spacing). 29 joined-word OCR warnings flagged (not auto-fixed).

## Tools

- `tools/pgs_to_srt.py` - PGS bitmap subtitle → SRT via Tesseract OCR
- `tools/extract_speakers.py` - Speaker attribution via SDH mining + rule-based + Claude Vision
- `tools/cleanup_transcript.py` - Format normalization across all transcript files
