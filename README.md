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

- `tools/extract_speakers.py` — Line merge + SDH mining + rule-based + batched Vision speaker attribution
- `tools/cleanup_transcript.py` — Format normalization (brackets, dashes, colon spacing, blank lines)
- `tools/pgs_to_srt.py` — PGS bitmap subtitle (Blu-ray) to SRT via Tesseract OCR
- `tools/whisperx_diarize.py` — Speaker diarization + voice embedding pipeline (see below)

## Diarization Pipeline

Unsupervised speaker identification using whisperX + ECAPA-TDNN voice embeddings.
Runs in a Docker container for GPU acceleration and reproducible dependencies.

### Quick start

```bash
# 1. Copy and edit docker-compose.example.yml with your video paths
cp docker-compose.example.yml docker-compose.yml

# 2. Build the container (one-time, ~5 min)
docker compose build

# 3. Check pipeline progress
docker compose run whisperx status

# 4. Process episodes (GPU-accelerated)
docker compose run whisperx process --season 1 --workers 1

# 5. Build speaker cluster reports for review
docker compose run whisperx embed-clusters --season 1
```

**Prerequisites:** Docker Desktop (WSL2 backend), NVIDIA GPU driver, [HuggingFace token](https://huggingface.co/pyannote/speaker-diarization-3.1) set as `HUGGINGFACE_TOKEN` env var.

See [docker-compose.example.yml](docker-compose.example.yml) for full setup notes.

## Format

Wiki transcript format:
- Two spaces after speaker colon: `Character:  dialogue`
- Scene descriptions in brackets: `[Scene description.]`
- Inline actions: `Character:  [gasps] Dialogue here.`
