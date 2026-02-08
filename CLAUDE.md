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
| `auto-label` | Propose/apply cluster labels using voice profiles (cosine similarity) | instant |
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

### Bulk workflow (once profiles are seeded)

**Important**: `--season N` matches across all series (AT, DL, FC). Use `--series AT` to scope to Adventure Time only.

```
1. process --series AT --season 1 --workers 1        # batch diarize (GPU)
2. embed-clusters --series AT --season 1             # batch clustering
3. auto-label --series AT --season 1                 # preview: compare clusters against profiles
4. auto-label --series AT --season 1 --apply         # write label files (no profile merge)
5. [Review + fix false positives in labels]          # Claude spot-checks reports, user approves
6. auto-label --series AT --season 1 --merge         # merge reviewed labels into voice profiles
7. validate --series AT --season 1                   # compare against transcripts
```

`auto-label` classifies each cluster centroid against voice profile centroids via cosine
similarity. Labels with `AUTO` (margin >= 0.05 over 2nd best) or `auto` (close margin).
Clusters below threshold are flagged `REVIEW` for manual identification.

- `--apply` writes label files only (no profile merge) — safe to re-run with `--force`
- `--merge` reads existing label files and merges into voice profiles — run after review
- Use both together (`--apply --merge`) to write + merge in one step (old behavior)
- `--additive` only processes clusters currently in `skipped`, preserving existing `speaker_map` entries. Use after manual review to expand coverage without overwriting corrections.
- **Temporal filtering**: `first_episode` and `last_episode` fields in `_index.json` per-profile prevent anachronistic matches. `first_episode` blocks early episodes (e.g. Fern profile from S09 won't match S01 clusters). `last_episode` blocks later episodes (e.g. AT Finn buckets and Prismo won't match FC/DL where voice actors differ). Applied automatically during auto-label.

Threshold system (layered):
- **Base threshold**: `--threshold` (default 0.60)
- **Sample-count scaling**: profiles with <100 samples get up to +0.10 penalty
- **Per-character override**: `--char-threshold "Ice King=0.65"` for known confusable voices (Tom Kenny voices many AT characters)

### Data layout

50 voice profiles, ~42K samples across S01-S10. 277 label files.

```
diarization/
  AT.S08E01.json              # v2 whisperX output (from process)
  clusters/AT.S08E01.npz      # per-cluster embeddings (from embed-clusters)
  labels/AT.S08E01.json       # cluster→character mapping (from embed-label/auto-label)
  reports/AT.S08E01.txt       # cluster ID report for review (from embed-clusters)
  voice_profiles/Finn.npz     # accumulated character profiles (from embed-label)
  voice_profiles/_index.json   # profile metadata
  validation_progress.json    # validate results with full disagree details per episode
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
The script is live-mounted (`.:/app`), so code changes are immediately available without rebuilding — only rebuild when dependencies change.

### Performance notes

- `process --device auto` auto-detects GPU; CUDA gives ~5x speedup over CPU
- Docker container uses Linux `fork` to share model memory via copy-on-write -- more workers from same RAM
- Windows native uses `spawn` (each worker loads its own model copy) -- 2 workers is the practical max
- `embed-clusters` loads ECAPA model once and processes episodes sequentially; omit `--episode` for auto-discovery
- `embed-label` is instant (numpy + JSON), safe to parallelize
- `auto-label` loads profiles once, compares centroids — no model loading, instant per episode
- `status` scans filesystem only, no model loading

### Known false positive patterns

- **Joshua (deep male voice)**: Consistently matches random deep-voiced background characters (Nightosphere demons, mud scamps). Hit 3 false positives in S04 alone. Sample-count penalty and temporal filtering help but don't fully prevent.
- **Young female voices (worst confusable cluster)**: Me-Mow, Young Marceline, Fionna, and Young PB profiles all cross-match with each other and with random young/feminine-sounding characters. Worse than the deep male voice problem — temporal bounds (`last_episode`) are more effective than threshold tuning for preventing cross-series false positives.
- **Elements arc transformations** (S09E02-E09): Characters get elementally transformed and their voice shifts enough to match *other* profiles (e.g. transformed Flame Princess → Marceline). All S09 Elements episodes need manual review.
- **Themed intros**: Special arcs (Elements, Islands) have custom intros sung by cast members. These cluster separately and match character profiles but aren't actual character dialogue — always false positives.
- **Alternate-world episodes**: Same voice actors play alternate versions of characters (e.g. Beyond the Grotto S10E03). Voice matches are technically correct but character identities differ. Accept or skip based on project needs.
- **~40% map rate is normal**: Across S01-S10 the mapping rate is consistently ~40%. The other 60% are intro/outro music, minor one-off characters, and mixed clusters. Don't chase 100%.
- **Mixed clusters**: When pyannote lumps multiple characters into one cluster (e.g. S01E01 SPEAKER_06 had Gumball Guardian + Chocoberry + Chet), skip the cluster rather than mislabel. Look for clusters where validation disagrees span many different transcript speakers.

### Profile contamination limits

Investigated whether surgical profile cleanup could improve Finn/Jake separation. Key findings:

- **Baseline cosine similarities**: Finn_S01-S03 ↔ Jake = 0.39, Finn ↔ PB = 0.30, Jake ↔ PB = 0.07
- **Finn/Jake contamination is pervasive**: 94% of Finn_S01-S03 episodes also have Jake. Only 5 Finn episodes are Jake-free. ~525 Finn↔Jake swaps across S01 validation.
- **Finn/PB contamination is localized**: Only 7 S01 episodes show Finn/PB swaps (87 total). Top 3 episodes account for 69% of it.
- **Removing contaminated episodes makes things worse**: Tested removing 5 worst Finn↔Jake episodes from both profiles. Similarity *increased* from 0.387 to 0.416 (+7%). Those episodes contribute far more correct samples (~80%) than contaminating ones (~20%), so removing them shrinks the profile and shifts the centroid toward noisier data.
- **Segment-level cleanup would require solving the original problem**: Can't identify which individual audio segments within a cluster belong to the wrong speaker without ground-truth diarization.
- **Conclusion**: The 0.39 Finn/Jake similarity is a fundamental property of the voices (young Jeremy Shada vs John DiMaggio), not a correctable artifact. Accept it as a known limitation.

### Key design decisions

- **No circular dependency**: Cluster embeddings come from anonymous pyannote speakers, not transcript labels
- **Pairwise split detection**: Cosine similarity >= 0.75 flags likely same-character splits
- **Season bucketing**: Finn gets separate profiles per season range (voice actor aging)
- **Alias resolution**: Uses the-enchiridion character data for canonical names. `_canon()` strips season-bucket suffixes (Finn_S01-S03 → Finn) and series-bucket suffixes (Prismo_FC → Prismo) then resolves aliases (Lumpy Space Princess → LSP) for comparison.
- **MANUAL_ALIASES vs `_canon()`**: `MANUAL_ALIASES` feeds `_resolve_speaker()` which is called during **profile creation** (embed-label line 2020). Adding bucket names here (e.g. `"Prismo_FC": "Prismo"`) causes profile contamination — FC samples merge into the AT profile. Series-bucket stripping belongs in `_canon()` only, which is used for **validation comparison** only. Never add bucket variants to MANUAL_ALIASES.
- **Profile rebuild procedure**: If a profile gets contaminated, delete the .npz, find label files with that character's mappings (`grep` labels/), re-run `embed-label` for each episode.
- **Skip mixed clusters**: When pyannote merges two characters, skip rather than mislabel
- **`last_episode` string comparison sorts series naturally**: Episode IDs sort alphabetically as `AT.*` < `DL.*` < `FC.*`, so `last_episode: "AT.S10E13"` blocks all DL/FC matches without series-aware logic. Convenient for this project's naming convention.

### Validate expected results

`validate` fuzzy-matches transcript lines to whisperX segments by timestamp, then compares speaker attributions using canonical name resolution.

- **~51% agree rate** is the S01 baseline after alias resolution. Remaining disagrees are real diarization errors, not transcript issues.
- **Finn↔Jake confusion**: 35% of all S01 disagrees. Pyannote struggles to separate two young male voices. Transcript is almost always correct.
- **Minor characters in major clusters**: 26% of disagrees. Characters without voice profiles land in the nearest major cluster. Expected and not fixable without seeding more profiles.
- **Finn→PB confusion**: Severe in close dialogue scenes (e.g. S01E01 cemetery scene: 26/68 disagrees). Young Finn and PB cluster together when alone.
- **Multi-speaker lines**: "Finn and Jake" etc. — inherently ambiguous, ~1% of disagrees.
- Validate is primarily useful for **spotting transcript attribution errors** and **measuring diarization quality**, not for correcting diarization.

## Video Archive

- Main series (BluRay): `D:\Shows\Adventure Time (2010) Season 1-10 S01-S10 + Extras (...)\`
- Distant Lands (WEB-DL): `S:\Shows\Adventure Time Distant Lands (2020) Season 1 S01 (...)\`
- F&C S01 (WEB-DL): `S:\Shows\Fionna and Cake (2023)\Season 1\`
- F&C S02 (WEB-DL): `D:\Shows\` (loose files, not in a season folder)

## Transcript Processing Gotchas

### ffmpeg

- Returns exit code 0 even when it fails to create a file — always check file existence AND size > 100 bytes after extraction
- **Multi-language releases**: WEB-DL files tagged "DUAL" may have a non-English audio track as the default stream. Always verify the default track language with `ffprobe -show_streams -select_streams a` before processing. Use `--audio-track N` to select the correct stream (e.g. `--audio-track 1` for second audio track).

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
