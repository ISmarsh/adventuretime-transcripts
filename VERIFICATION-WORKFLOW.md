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

### 6. Audio Speaker Diarization (pyannote)

For episodes where visual verification is difficult (e.g. puppet shows, off-screen
dialogue, or allegorical retellings), speaker diarization can cluster voices from the
audio track.

#### Setup

Requires Python packages: `pyannote.audio`, `faster-whisper`, `torch`, `soundfile`

```bash
pip install pyannote.audio faster-whisper soundfile
```

Requires a HuggingFace token with accepted terms for:
- `pyannote/speaker-diarization-3.1`
- `pyannote/segmentation-3.0`

Set `HF_TOKEN` environment variable with a Read-scope token.

#### Extract audio

```bash
ffmpeg -y -i "<VIDEO_FILE>" -vn -acodec pcm_s16le -ar 16000 -ac 1 "<OUTPUT>.wav"
```

#### Run diarization

```python
import torch

# Monkey-patch for PyTorch 2.8+ compatibility
_original = torch.load
def _patched(*args, **kwargs):
    kwargs["weights_only"] = False
    return _original(*args, **kwargs)
torch.load = _patched

from pyannote.audio import Pipeline

pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token=os.environ["HF_TOKEN"],
)
pipeline.to(torch.device("cpu"))
diarization = pipeline("audio.wav")

for turn, _, speaker in diarization.itertracks(yield_label=True):
    print(f"[{turn.start:.1f}-{turn.end:.1f}] {speaker}")
```

#### Mapping to transcript

Match diarization segments to transcript lines by timestamp overlap. Identify
speaker clusters using known reference lines (e.g. a line you're certain about).

#### What works well

- Speaker separation is highly accurate — in testing on S09E11 Ketchup, pyannote
  agreed with manual corrections on nearly every line (only missed one short
  interjection out of ~25 changes)
- Distinguishing 2-3 main speakers in dialogue scenes
- Animated voices with distinct pitch/timbre (e.g. BMO vs Marceline)
- Finding speaker boundaries when lines are misattributed
- More reliable than contextual/LLM analysis for determining who is speaking

#### Known limitations

- Singing voices may cluster separately from speaking voices
- Character impressions / puppet voices can confuse clustering
- Short exclamations ("Whoa!", "Huh?") may not cluster reliably
- CPU-only runs take ~2-5 min per 11-min episode
- PyTorch 2.8+ requires the `weights_only=False` monkey-patch shown above
- Windows without Developer Mode needs workarounds for HuggingFace symlinks

#### When to use

Best for episodes with ambiguous speaker attribution that can't be resolved by
SDH subtitles or frame extraction alone. Especially useful for:
- Two-character bottle episodes
- Narrated/retold sequences where the narrator isn't the character shown
- Scenes where characters are off-screen

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
