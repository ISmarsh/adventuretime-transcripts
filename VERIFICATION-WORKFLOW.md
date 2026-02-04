# Transcript Speaker Verification Workflow

Verifying speaker attribution in SRT-converted transcripts before wiki submission.

## Problem

SRT files lack speaker labels. When converting to wiki format, attribution is guessed from context. Common errors:

- Paired character swaps (two characters sharing a scene)
- Lines split incorrectly at scene boundaries
- Third-person references misattributed

## Wiki Format

- Two spaces after character colon: `Character:  dialogue`
- Scene descriptions in brackets: `[Scene description here.]`
- Inline actions: `Character:  [gasps] Dialogue here.`

## Verification Steps

### 1. Extract SDH Subtitles

SDH (Subtitles for the Deaf/Hard of Hearing) often include speaker labels.

Check subtitle streams:
```bash
ffprobe -v error -show_entries stream=index,codec_type:stream_tags=title -of csv=p=0 "<VIDEO_FILE>"
```

Extract SDH track (often index 3):
```bash
ffmpeg -y -i "<VIDEO_FILE>" -map 0:<SDH_INDEX> "<OUTPUT>.srt"
```

### 2. Search SDH for Speaker Labels

```bash
grep -i "<KEYWORD>" "<SRT_FILE>"
```

SDH labels may appear as `[CHARACTER]` or `CHARACTER:` before dialogue.

### 3. Extract Video Frames

For lines without SDH labels, extract frames around the timestamp:

```bash
# Extract multiple frames to catch mouth animation
ffmpeg -y -ss <TIMESTAMP> -i "<VIDEO_FILE>" -vframes 1 -q:v 2 "<OUTPUT>_1.jpg"
ffmpeg -y -ss <TIMESTAMP+1s> -i "<VIDEO_FILE>" -vframes 1 -q:v 2 "<OUTPUT>_2.jpg"
ffmpeg -y -ss <TIMESTAMP+2s> -i "<VIDEO_FILE>" -vframes 1 -q:v 2 "<OUTPUT>_3.jpg"
```

### 4. Check Mouth Animation

Extract 2-3 frames spanning the line's duration. Look for:
- Which character has mouth open/animated
- Who's on screen vs off screen
- Reaction shots (listener vs speaker)

A single frame may catch a closed mouth mid-word. Multiple frames give better coverage.

### 5. Record Uncertain Moments

Track lines that can't be definitively verified for later review:

```
Episode @ timestamp - "Line text"
  Current: CharacterA
  Analysis: CharacterB visible with mouth animation
  Verdict: UNCERTAIN or FIX NEEDED
```

## Common Error Patterns

### Paired Character Swaps
When two characters share a scene, their lines often get swapped.

### Scene Boundary Errors
One scene may end and another begin mid-dialogue block. Check:
- Timestamps for gaps
- Background/setting changes
- Music transitions

### Speech Markers
Some characters have distinctive speech patterns (catchphrases, verbal tics) that help identify them even without video.
