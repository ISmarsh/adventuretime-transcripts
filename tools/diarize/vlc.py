"""VLC clip extraction and playback utilities."""

import ctypes
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

CLIP_PADDING = 0.2  # seconds of padding before/after each clip

VLC_PATHS = [
    "C:/Program Files/VideoLAN/VLC/vlc.exe",
    "C:/Program Files (x86)/VideoLAN/VLC/vlc.exe",
]


def find_vlc() -> str | None:
    """Find VLC executable."""
    path = shutil.which("vlc")
    if path:
        return path
    for p in VLC_PATHS:
        if Path(p).exists():
            return p
    return None


def extract_clip(
    video_path: str, start: float, end: float, output_path: str,
    audio_track: int | None = None,
) -> bool:
    """Extract a short video clip using ffmpeg stream copy."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return False

    padded_start = max(0, start - CLIP_PADDING)
    padded_end = end + CLIP_PADDING

    duration = padded_end - padded_start
    cmd = [
        ffmpeg, "-y",
        "-ss", str(padded_start),
        "-i", video_path,
        "-t", str(duration),
    ]
    if audio_track is not None:
        cmd += ["-map", "0:v:0", "-map", f"0:a:{audio_track}"]
    cmd += ["-c", "copy", "-avoid_negative_ts", "1", output_path]

    result = subprocess.run(cmd, capture_output=True, timeout=30)
    return (
        result.returncode == 0
        and Path(output_path).exists()
        and Path(output_path).stat().st_size > 100
    )


def play_clips_vlc(vlc_path: str, clip_paths: list[str]) -> subprocess.Popen:
    """Play clips in VLC (non-blocking). Returns process handle."""
    if sys.platform == "win32":
        user32 = ctypes.windll.user32
        hwnd = user32.GetForegroundWindow()
    else:
        hwnd = None

    proc = subprocess.Popen(
        [vlc_path, "--play-and-exit", "--no-repeat", "--no-loop"] + clip_paths,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    if sys.platform == "win32" and hwnd:
        def _refocus():
            for _ in range(5):
                time.sleep(0.3)
                if hwnd:
                    user32.keybd_event(0x12, 0, 0, 0)   # ALT down
                    user32.SetForegroundWindow(hwnd)
                    user32.keybd_event(0x12, 0, 2, 0)   # ALT up

        threading.Thread(target=_refocus, daemon=True).start()

    return proc


def stop_vlc(proc: subprocess.Popen | None) -> None:
    """Kill a running VLC process if still alive."""
    if proc and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()
