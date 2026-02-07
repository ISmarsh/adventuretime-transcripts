# Adventure Time Transcripts

282 transcript files across Adventure Time, Distant Lands, and Fionna & Cake.

See also: [README.md](README.md) (format standard, project overview),
[PGS-OCR-WORKFLOW.md](PGS-OCR-WORKFLOW.md) (BluRay subtitle extraction),
[VERIFICATION-WORKFLOW.md](VERIFICATION-WORKFLOW.md) (SDH extraction, speaker verification),
[TRANSCRIPT-CORRECTIONS.md](TRANSCRIPT-CORRECTIONS.md) (OCR join fixes, Vision cleanup stats),
[TRANSCRIPT-GAPS.md](TRANSCRIPT-GAPS.md) (episode fill sources)

## Tools

- `tools/extract_speakers.py` — line merge + SDH mining + rule-based + batched Vision attribution
- `tools/cleanup_transcript.py` — format normalization (brackets, dashes, colon spacing, blank lines)
- `tools/pgs_to_srt.py` — PGS bitmap subtitle → SRT via Tesseract OCR
- `tools/whisperx_diarize.py` — speaker diarization + voice embedding pipeline (see below)

## Diarization Pipeline

Unsupervised speaker identification using whisperX + ECAPA-TDNN embeddings.
Transcripts are NOT used as ground truth — they are the thing being validated.

### Subcommands

| Command | Purpose | Time |
|---------|---------|------|
| `process` | whisperX transcription + pyannote diarization → v2 JSON | ~8 min/ep (CPU), ~1-2 min (GPU) |
| `embed-clusters` | Build per-cluster embeddings, print ID report, save to reports/ | ~20s/ep |
| `embed-label` | Save cluster→character mapping, merge into voice profiles | instant |
| `validate` | Compare diarization against transcript (uses label files when available) | ~20s |
| `status` | Show pipeline progress across all episodes | instant |

### Workflow per episode

```
1. process --episode AT.S08E08           # generate v2 JSON with speaker clusters
2. embed-clusters --episode AT.S08E08    # print cluster report with sample dialogue
3. [Claude identifies characters from dialogue + show knowledge]
4. [User confirms or corrects mapping]
5. embed-label --episode AT.S08E08 --map SPEAKER_00=Starchy SPEAKER_01=Finn ...
6. validate --episode AT.S08E08          # compare against transcript
```

### Data layout

```
diarization/
  AT.S08E01.json              # v2 whisperX output (from process)
  clusters/AT.S08E01.npz      # per-cluster embeddings (from embed-clusters)
  labels/AT.S08E01.json       # cluster→character mapping (from embed-label)
  reports/AT.S08E01.txt       # cluster ID report for review (from embed-clusters)
  voice_profiles/Finn.npz     # accumulated character profiles (from embed-label)
  voice_profiles/_index.json   # profile metadata
```

### Docker setup (GPU processing)

For bulk processing with GPU acceleration, use the Docker container:

```bash
docker compose build                                      # one-time (~5 min)
docker compose run whisperx status                        # check pipeline progress
docker compose run whisperx process --season 1 --workers 1  # process with GPU
docker compose run whisperx embed-clusters --season 1     # batch clustering
```

Requires: Docker Desktop (WSL2 backend), NVIDIA GPU driver, `HF_TOKEN` env var.
Video dirs mounted read-only from host; diarization output writes directly to project.

### Performance notes

- `process --device auto` auto-detects GPU; CUDA gives ~5x speedup over CPU
- Docker container uses Linux `fork` to share model memory via copy-on-write -- more workers from same RAM
- Windows native uses `spawn` (each worker loads its own model copy) -- 2 workers is the practical max
- `embed-clusters` loads ECAPA model once and processes episodes sequentially; omit `--episode` for auto-discovery
- `embed-label` is instant (numpy + JSON), safe to parallelize
- `status` scans filesystem only, no model loading

### Key design decisions

- **No circular dependency**: Cluster embeddings come from anonymous pyannote speakers, not transcript labels
- **Pairwise split detection**: Cosine similarity >= 0.75 flags likely same-character splits
- **Season bucketing**: Finn gets separate profiles per season range (voice actor aging)
- **Alias resolution**: Uses the-enchiridion character data for canonical names
- **Skip mixed clusters**: When pyannote merges two characters, skip rather than mislabel

## Video Archive

- Main series (BluRay): `D:\Shows\Adventure Time (2010) Season 1-10 S01-S10 + Extras (...)\`
- Distant Lands (WEB-DL): `S:\Shows\Adventure Time Distant Lands (2020) Season 1 S01 (...)\`
- F&C S01 (WEB-DL): `S:\Shows\Fionna and Cake (2023)\Season 1\`
- F&C S02 (WEB-DL): `D:\Shows\` (loose files, not in a season folder)

## Transcript Processing Gotchas

### ffmpeg

- Returns exit code 0 even when it fails to create a file — always check file existence AND size > 100 bytes after extraction

### Vision API (Haiku)

- Rate limit: ~50K input tokens/min — run episodes sequentially, not in parallel
- Retry backoff: 10s, 20s, 40s, 80s, 160s handles 429s well
- Frame dedup via MD5 saves ~30% tokens; scene batching cuts per-line cost ~45%

### Windows / Python

- Python stdout defaults to cp1252 on Windows — force UTF-8 with `io.TextIOWrapper`

### SDH / Subtitle Sources

- F&C WEB-DL SDH has no speaker names (just sound effects like `[Screaming]`) — useless for attribution
- BluRay PGS: bitmap subs, need OCR — see [PGS-OCR-WORKFLOW.md](PGS-OCR-WORKFLOW.md)
- SDH extraction commands and speaker label search: [VERIFICATION-WORKFLOW.md](VERIFICATION-WORKFLOW.md)
- OCR join splitting methodology: [TRANSCRIPT-CORRECTIONS.md](TRANSCRIPT-CORRECTIONS.md)
