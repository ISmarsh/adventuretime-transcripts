"""ECAPA-TDNN model management, audio extraction, and embedding computation."""

import logging
import subprocess
import sys
from pathlib import Path

from .config import ECAPA_SAVEDIR, ECAPA_SOURCE
from .discovery import find_tool

logger = logging.getLogger("whisperx_diarize")

_ecapa_model = None


def _get_ecapa():
    """Lazy-load SpeechBrain ECAPA-TDNN model."""
    global _ecapa_model  # noqa: PLW0603
    if _ecapa_model is None:
        logging.getLogger("speechbrain").setLevel(logging.WARNING)
        from speechbrain.inference.speaker import EncoderClassifier
        from speechbrain.utils.fetching import LocalStrategy
        kwargs = {
            "source": ECAPA_SOURCE,
            "savedir": ECAPA_SAVEDIR,
            "run_opts": {"device": "cpu"},
        }
        # Windows: symlinks require admin, use COPY instead
        if sys.platform == "win32":
            kwargs["local_strategy"] = LocalStrategy.COPY
        _ecapa_model = EncoderClassifier.from_hparams(**kwargs)
    return _ecapa_model


def _extract_wav(
    video_path: Path, output_path: Path, audio_track: int | None = None,
) -> bool:
    """Extract 16kHz mono WAV from video. Returns True on success."""
    cmd = [find_tool("ffmpeg"), "-y", "-i", str(video_path)]
    if audio_track is not None:
        cmd += ["-map", f"0:a:{audio_track}"]
    cmd += ["-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
            str(output_path)]
    subprocess.run(cmd, capture_output=True)
    return output_path.exists() and output_path.stat().st_size > 100


def _load_wav(wav_path: Path):
    """Load WAV file. Returns (waveform, sample_rate)."""
    import torchaudio
    return torchaudio.load(str(wav_path))


def _slice_segment(waveform, sr: int, start: float, end: float):
    """Slice an audio segment from the full waveform."""
    s = int(start * sr)
    e = min(int(end * sr), waveform.shape[1])
    return waveform[:, s:e]


def _compute_embeddings_batch(segments: list):
    """Compute ECAPA-TDNN embeddings for audio tensors. Returns (N, 192) ndarray."""
    import torch

    model = _get_ecapa()
    max_len = max(s.shape[1] for s in segments)
    batch = torch.zeros(len(segments), max_len)
    wav_lens = torch.zeros(len(segments))
    for i, seg in enumerate(segments):
        batch[i, :seg.shape[1]] = seg[0]
        wav_lens[i] = seg.shape[1] / max_len

    with torch.no_grad():
        embeddings = model.encode_batch(batch, wav_lens)

    return embeddings.squeeze(1).numpy()


def _filter_outliers(embeddings, threshold_std: float = 2.0):
    """Return boolean mask of embeddings to keep (remove outliers)."""
    import numpy as np
    if len(embeddings) < 4:
        return np.ones(len(embeddings), dtype=bool)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normed = embeddings / np.maximum(norms, 1e-10)
    sim_matrix = normed @ normed.T
    mean_sim = sim_matrix.mean(axis=1)
    threshold = mean_sim.mean() - threshold_std * mean_sim.std()
    return mean_sim >= threshold


def _compute_centroid(embeddings):
    """Compute L2-normalized centroid from embeddings with outlier filtering."""
    import numpy as np
    keep = _filter_outliers(embeddings)
    filtered = embeddings[keep]
    if len(filtered) == 0:
        filtered = embeddings
    centroid = filtered.mean(axis=0)
    norm = np.linalg.norm(centroid)
    if norm > 0:
        centroid /= norm
    return centroid
