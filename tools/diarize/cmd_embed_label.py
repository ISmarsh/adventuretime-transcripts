"""Embed-label subcommand: save cluster->character mapping and merge into profiles."""

import argparse
import io
import json
import sys
from datetime import datetime
from pathlib import Path

from .config import PROJECT_ROOT
from .embeddings import _compute_centroid
from .profiles import _load_profile, _profile_dir, _rebuild_index, _save_profile
from .speakers import _get_profile_name, _resolve_speaker


def cmd_embed_label(args: argparse.Namespace) -> None:
    """Save cluster->character mapping and merge into character profiles."""
    import numpy as np

    if sys.stdout.encoding != "utf-8":
        sys.stdout = io.TextIOWrapper(
            sys.stdout.buffer, encoding="utf-8", errors="replace")

    dia_dir = PROJECT_ROOT / args.diarization_dir
    pdir = _profile_dir(dia_dir)

    ep_id = args.episode

    # Parse --map args: SPEAKER_00=Finn SPEAKER_01=Jake ...
    speaker_map: dict[str, str] = {}
    skips: set[str] = set()
    if args.skip:
        skips = set(args.skip)

    for mapping in args.map:
        if "=" not in mapping:
            print(f"Error: invalid mapping '{mapping}' — "
                  f"use CLUSTER=Character format", file=sys.stderr)
            sys.exit(1)
        cluster, character = mapping.split("=", 1)
        canonical = _resolve_speaker(character)
        speaker_map[cluster] = canonical

    if not speaker_map:
        print("Error: no mappings provided", file=sys.stderr)
        sys.exit(1)

    # Load cluster embeddings
    cluster_path = dia_dir / "clusters" / f"{ep_id}.npz"
    if not cluster_path.exists():
        print(f"Error: {cluster_path} not found. "
              f"Run 'embed-clusters --episode {ep_id}' first.",
              file=sys.stderr)
        sys.exit(1)

    cluster_data = np.load(str(cluster_path), allow_pickle=False)
    meta_str = str(cluster_data.get("_meta", "{}"))
    meta = json.loads(meta_str)
    season = meta.get("season", 0)

    print(f"Episode: {ep_id} ({meta.get('title', '')})")
    print(f"Mapping {len(speaker_map)} clusters:")
    for cluster, character in sorted(speaker_map.items()):
        print(f"  {cluster} -> {character}")
    if skips:
        print(f"Skipping: {', '.join(sorted(skips))}")

    # Merge cluster embeddings into character profiles
    for cluster, character in sorted(speaker_map.items()):
        if cluster in skips:
            continue

        emb_key = f"{cluster}_embeddings"
        if emb_key not in cluster_data:
            print(f"  Warning: no embeddings for {cluster}")
            continue

        new_embeddings = cluster_data[emb_key]
        if len(new_embeddings) == 0:
            print(f"  {cluster} -> {character}: no embeddings to merge")
            continue

        profile_name = _get_profile_name(character, season)
        profile_path = pdir / f"{profile_name}.npz"

        # Build metadata for new samples
        new_metadata = [{
            "episode": ep_id,
            "season": season,
            "cluster": cluster,
            "sample_idx": i,
        } for i in range(len(new_embeddings))]

        if profile_path.exists():
            existing = _load_profile(profile_path)
            all_embeddings = np.vstack(
                [existing["embeddings"], new_embeddings])
            all_metadata = existing["metadata"] + new_metadata
            print(f"  {cluster} -> {profile_name}: "
                  f"+{len(new_embeddings)} samples "
                  f"(total {len(all_embeddings)})")
        else:
            all_embeddings = new_embeddings
            all_metadata = new_metadata
            print(f"  {cluster} -> {profile_name}: "
                  f"{len(new_embeddings)} samples (new)")

        centroid = _compute_centroid(all_embeddings)
        pdir.mkdir(parents=True, exist_ok=True)
        _save_profile(profile_path, centroid, all_embeddings, all_metadata)

    # Save label file (merge into existing if present)
    labels_dir = dia_dir / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    label_path = labels_dir / f"{ep_id}.json"

    if label_path.exists() and not getattr(args, "replace", False):
        label_data = json.loads(label_path.read_text(encoding="utf-8"))
        label_data["speaker_map"].update(speaker_map)
        # Remove newly-mapped speakers from skipped list
        existing_skipped = set(label_data.get("skipped", []))
        existing_skipped -= set(speaker_map.keys())
        existing_skipped |= skips
        label_data["skipped"] = sorted(existing_skipped)
        label_data["timestamp"] = datetime.now().isoformat(timespec="seconds")
    else:
        label_data = {
            "episode_id": ep_id,
            "title": meta.get("title", ""),
            "season": season,
            "speaker_map": speaker_map,
            "skipped": sorted(skips),
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }

    label_path.write_text(
        json.dumps(label_data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\nSaved label mapping to {label_path}")

    n_profiles = _rebuild_index(pdir)
    print(f"Updated index: {n_profiles} profiles")
