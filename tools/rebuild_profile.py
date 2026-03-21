"""One-off script to rebuild a voice profile from existing label files.

Usage:
    python -m tools.rebuild_profile "Ice King"
    python -m tools.rebuild_profile "Ice King" --dry-run

Scans all label files for the given character, loads cluster embeddings,
and rebuilds the profile .npz from scratch without touching label files.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tools.diarize.config import PROJECT_ROOT as PROJ
from tools.diarize.embeddings import _compute_centroid
from tools.diarize.profiles import _profile_dir, _rebuild_index, _save_profile
from tools.diarize.speakers import _get_profile_name


def main():
    parser = argparse.ArgumentParser(description="Rebuild a voice profile from labels")
    parser.add_argument("character", help="Character name (e.g. 'Ice King')")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done")
    parser.add_argument("--diarization-dir", default="diarization",
                        help="Diarization directory (default: diarization)")
    args = parser.parse_args()

    dia_dir = PROJ / args.diarization_dir
    labels_dir = dia_dir / "labels"
    clusters_dir = dia_dir / "clusters"
    pdir = _profile_dir(dia_dir)

    character = args.character

    # Scan all label files for this character
    episodes = []
    for label_path in sorted(labels_dir.glob("*.json")):
        data = json.loads(label_path.read_text(encoding="utf-8"))
        speaker_map = data.get("speaker_map", {})
        season = data.get("season", 0)
        ep_id = data.get("episode_id", label_path.stem)

        matching_clusters = [
            cluster for cluster, char in speaker_map.items()
            if char == character
        ]
        if matching_clusters:
            episodes.append((ep_id, season, matching_clusters))

    if not episodes:
        print(f"No label files found mapping to '{character}'")
        sys.exit(1)

    print(f"Rebuilding profile for: {character}")
    print(f"Found {len(episodes)} episodes with {character} labels")

    # Collect all embeddings
    all_embeddings = []
    all_metadata = []
    missing_clusters = 0
    profile_name = None

    for ep_id, season, clusters in episodes:
        cluster_path = clusters_dir / f"{ep_id}.npz"
        if not cluster_path.exists():
            print(f"  {ep_id}: cluster file missing, skipping")
            missing_clusters += 1
            continue

        cluster_data = np.load(str(cluster_path), allow_pickle=False)

        if profile_name is None:
            profile_name = _get_profile_name(character, season)

        for cluster in clusters:
            emb_key = f"{cluster}_embeddings"
            if emb_key not in cluster_data:
                print(f"  {ep_id} {cluster}: no embeddings, skipping")
                continue

            embeddings = cluster_data[emb_key]
            if len(embeddings) == 0:
                continue

            metadata = [{
                "episode": ep_id,
                "season": season,
                "cluster": cluster,
                "sample_idx": i,
            } for i in range(len(embeddings))]

            all_embeddings.append(embeddings)
            all_metadata.extend(metadata)

    if not all_embeddings:
        print("No embeddings found!")
        sys.exit(1)

    combined = np.vstack(all_embeddings)
    centroid = _compute_centroid(combined)

    print(f"\nProfile: {profile_name}")
    print(f"  Episodes: {len(episodes)} ({missing_clusters} missing cluster files)")
    print(f"  Total samples: {len(combined)}")
    print(f"  Embedding dim: {combined.shape[1]}")

    if args.dry_run:
        print("\n[DRY RUN] Would save to:", pdir / f"{profile_name}.npz")
        return

    profile_path = pdir / f"{profile_name}.npz"
    if profile_path.exists():
        print(f"\n  WARNING: {profile_path.name} already exists, overwriting")

    _save_profile(profile_path, centroid, combined, all_metadata)
    print(f"  Saved: {profile_path}")

    n_profiles = _rebuild_index(pdir)
    print(f"  Index updated: {n_profiles} profiles")


if __name__ == "__main__":
    main()
