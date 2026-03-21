"""Embed-clusters subcommand: build per-cluster ECAPA-TDNN embeddings."""

import argparse
import io
import json
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

from .config import (
    DEFAULT_VIDEO_DIRS,
    EMBED_DIM,
    MIN_SLICE_DURATION,
    PROGRESS_FILE,
    PROJECT_ROOT,
    SPLIT_SIM_POSSIBLE,
    SPLIT_SIM_THRESHOLD,
)
from .discovery import (
    discover_transcripts,
    format_duration,
    match_episodes,
    scan_videos,
)
from .embeddings import (
    _compute_centroid,
    _compute_embeddings_batch,
    _extract_wav,
    _get_ecapa,
    _load_wav,
    _slice_segment,
)


def _embed_clusters_one(
    ep_id: str, data: dict, video_path: Path, dia_dir: Path, args: argparse.Namespace,
) -> str:
    """Build per-cluster embeddings for a single episode. Returns report text."""
    import numpy as np

    segments = data.get("segments", [])
    by_cluster: dict[str, list[dict]] = defaultdict(list)
    for seg in segments:
        spk = seg.get("speaker", "")
        if spk:
            by_cluster[spk].append(seg)

    lines: list[str] = []
    lines.append(f"Episode: {ep_id} ({data.get('title', '')})")
    lines.append(f"{len(by_cluster)} speaker clusters, {len(segments)} total segments\n")

    tmp_dir = tempfile.mkdtemp(prefix="embed_clusters_")
    wav_path = Path(tmp_dir) / f"{ep_id}.wav"

    try:
        if not _extract_wav(video_path, wav_path,
                            getattr(args, "audio_track", None)):
            lines.append("ERROR: WAV extraction failed")
            return "\n".join(lines)

        waveform, sr = _load_wav(wav_path)

        cluster_centroids = {}
        cluster_embeddings = {}
        cluster_meta = {}

        for spk_id in sorted(by_cluster.keys()):
            segs = by_cluster[spk_id]

            eligible = []
            for seg in segs:
                dur = seg["end"] - seg["start"]
                if args.min_duration <= dur <= args.max_duration:
                    eligible.append(seg)

            total_time = sum(s["end"] - s["start"] for s in segs)

            audio_segments = []
            valid_segs = []
            for seg in eligible:
                sliced = _slice_segment(waveform, sr, seg["start"], seg["end"])
                if sliced.shape[1] >= sr * MIN_SLICE_DURATION:
                    audio_segments.append(sliced)
                    valid_segs.append(seg)

            if audio_segments:
                embs = _compute_embeddings_batch(audio_segments)
                centroid = _compute_centroid(embs)
                cluster_centroids[spk_id] = centroid
                cluster_embeddings[spk_id] = embs
            else:
                embs = np.zeros((0, EMBED_DIM), dtype=np.float32)
                cluster_embeddings[spk_id] = embs

            samples = []
            for seg in segs[:args.samples]:
                text = seg.get("text", "").strip()
                if text:
                    samples.append(text)

            cluster_meta[spk_id] = {
                "total_segments": len(segs),
                "embedded_segments": len(valid_segs),
                "total_time": round(total_time, 1),
                "samples": samples,
            }

            lines.append(f"\n{spk_id} ({len(segs)} segs, {total_time:.1f}s, "
                         f"{len(valid_segs)} embedded)")
            for s in samples:
                lines.append(f'  "{s[:80]}"')

        # Pairwise cluster similarity
        if len(cluster_centroids) >= 2:
            names = sorted(cluster_centroids.keys())
            centroids_arr = np.stack([cluster_centroids[n] for n in names])
            c_norms = np.linalg.norm(centroids_arr, axis=1, keepdims=True)
            c_normed = centroids_arr / np.maximum(c_norms, 1e-10)
            sim_matrix = c_normed @ c_normed.T

            lines.append(f"\n{'=' * 60}")
            lines.append("PAIRWISE SIMILARITY:")
            pairs = []
            for i in range(len(names)):
                for j in range(i + 1, len(names)):
                    sim = float(sim_matrix[i, j])
                    pairs.append((names[i], names[j], sim))
            pairs.sort(key=lambda x: -x[2])

            has_likely_split = False
            for a, b, sim in pairs:
                if sim >= SPLIT_SIM_THRESHOLD:
                    has_likely_split = True
                    lines.append(f"  {a} <-> {b}: {sim:.3f}  <- likely same character")
                elif sim >= SPLIT_SIM_POSSIBLE:
                    lines.append(f"  {a} <-> {b}: {sim:.3f}  <- possible match")

            if not has_likely_split:
                for a, b, sim in pairs[:3]:
                    lines.append(f"  {a} <-> {b}: {sim:.3f}")

        # Save cluster embeddings
        cluster_dir = dia_dir / "clusters"
        cluster_dir.mkdir(parents=True, exist_ok=True)
        save_path = cluster_dir / f"{ep_id}.npz"

        arrays = {}
        for spk_id in sorted(cluster_embeddings.keys()):
            arrays[f"{spk_id}_embeddings"] = cluster_embeddings[spk_id]
            if spk_id in cluster_centroids:
                arrays[f"{spk_id}_centroid"] = cluster_centroids[spk_id]

        arrays["_meta"] = json.dumps({
            "episode_id": ep_id,
            "title": data.get("title", ""),
            "season": data.get("season", 0),
            "clusters": cluster_meta,
        })

        np.savez_compressed(str(save_path), **arrays)
        lines.append(f"\nSaved cluster embeddings to {save_path}")

    finally:
        if wav_path.exists():
            wav_path.unlink(missing_ok=True)
        try:
            Path(tmp_dir).rmdir()
        except OSError:
            pass

    return "\n".join(lines)


def cmd_embed_clusters(args: argparse.Namespace) -> None:
    """Build per-cluster embeddings from anonymous speakers and print report."""
    if sys.stdout.encoding != "utf-8":
        sys.stdout = io.TextIOWrapper(
            sys.stdout.buffer, encoding="utf-8", errors="replace")

    dia_dir = PROJECT_ROOT / args.diarization_dir
    force = getattr(args, "force", False)
    limit = getattr(args, "limit", None)

    # Discover eligible episodes
    if args.episode:
        # Specific episodes requested
        ep_ids = args.episode if isinstance(args.episode, list) else [args.episode]
    else:
        # Auto-discover: have JSON, missing NPZ (unless --force)
        ep_ids = []
        cluster_dir = dia_dir / "clusters"
        existing_npz = (
            {p.stem for p in cluster_dir.glob("*.npz")} if cluster_dir.is_dir() else set()
        )
        for jp in sorted(dia_dir.glob("*.json")):
            if jp.name == PROGRESS_FILE:
                continue
            eid = jp.stem
            if force or eid not in existing_npz:
                ep_ids.append(eid)

    # Apply series/season filters
    series_filter = getattr(args, "series", "all")
    season_filter = getattr(args, "season", None)
    if series_filter and series_filter != "all":
        code = series_filter.upper()
        ep_ids = [e for e in ep_ids if e.startswith(code + ".")]
    if season_filter is not None:
        pattern = f"S{season_filter:02d}E"
        ep_ids = [e for e in ep_ids if pattern in e]

    if limit:
        ep_ids = ep_ids[:limit]

    if not ep_ids:
        print("No episodes to process.")
        return

    # Resolve video paths
    video_dirs = getattr(args, "video_dirs", DEFAULT_VIDEO_DIRS)
    video_index = scan_videos(video_dirs)
    all_transcripts = discover_transcripts(PROJECT_ROOT)
    match_episodes(all_transcripts, video_index)
    transcript_map = {ep.episode_id: ep for ep in all_transcripts}

    # Load ECAPA model once before the loop
    print("Loading ECAPA-TDNN model...")
    _get_ecapa()

    # Create report directory
    report_dir = dia_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    total = len(ep_ids)
    succeeded = 0
    t0 = time.monotonic()

    for idx, ep_id in enumerate(ep_ids, 1):
        json_path = dia_dir / f"{ep_id}.json"
        if not json_path.exists():
            print(f"[{idx:03d}/{total}] {ep_id} -- SKIP (no JSON)", file=sys.stderr)
            continue

        data = json.loads(json_path.read_text(encoding="utf-8"))
        segments = data.get("segments", [])
        if not segments:
            print(f"[{idx:03d}/{total}] {ep_id} -- SKIP (no segments)", file=sys.stderr)
            continue

        # Resolve video path: try JSON, then transcript match, then video index
        video_path = data.get("video_path", "")
        if not video_path or not Path(video_path).exists():
            ep = transcript_map.get(ep_id)
            if ep and ep.video_path:
                video_path = str(ep.video_path)

        if not video_path or not Path(video_path).exists():
            print(f"[{idx:03d}/{total}] {ep_id} -- SKIP (no video)", file=sys.stderr)
            continue

        elapsed = time.monotonic() - t0
        title = data.get("title", "")
        print(f"\n[{idx:03d}/{total}] {ep_id} {title}")

        report = _embed_clusters_one(ep_id, data, Path(video_path), dia_dir, args)
        print(report)

        # Save report file
        report_path = report_dir / f"{ep_id}.txt"
        report_path.write_text(report, encoding="utf-8")
        succeeded += 1

    elapsed = time.monotonic() - t0
    print(f"\n{'=' * 60}")
    print(f"Embed-clusters complete: {succeeded}/{total} in {format_duration(elapsed)}")
