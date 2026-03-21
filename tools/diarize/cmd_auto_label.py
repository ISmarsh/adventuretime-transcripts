"""Auto-label subcommand: classify clusters against voice profiles."""

import argparse
import io
import json
import sys
from datetime import datetime

from .config import AUTO_LABEL_MARGIN, PROJECT_ROOT, SMALL_PROFILE_PENALTY, SMALL_PROFILE_SAMPLES
from .embeddings import _compute_centroid
from .profiles import (
    _load_all_profiles,
    _load_profile,
    _load_profile_sample_counts,
    _load_profile_series_ranges,
    _profile_dir,
    _rebuild_index,
    _save_profile,
    _select_season_profiles,
)
from .speakers import _get_profile_name


def cmd_auto_label(args: argparse.Namespace) -> None:
    """Propose or apply cluster->character mappings using voice profiles."""
    import numpy as np

    if sys.stdout.encoding != "utf-8":
        sys.stdout = io.TextIOWrapper(
            sys.stdout.buffer, encoding="utf-8", errors="replace")

    dia_dir = PROJECT_ROOT / args.diarization_dir
    pdir = _profile_dir(dia_dir)
    threshold = args.threshold
    apply = args.apply
    merge = args.merge
    additive = getattr(args, "additive", False)

    # Parse per-character threshold overrides
    char_thresholds: dict[str, float] = {}
    if getattr(args, "char_threshold", None):
        for ct in args.char_threshold:
            if "=" not in ct:
                print(f"Error: invalid char-threshold '{ct}' — "
                      f"use Character=threshold format", file=sys.stderr)
                sys.exit(1)
            name, val = ct.split("=", 1)
            char_thresholds[name] = float(val)

    # Load voice profiles
    all_profiles = _load_all_profiles(pdir)
    if not all_profiles:
        print("No voice profiles found. Label some episodes with "
              "'embed-label' first.", file=sys.stderr)
        sys.exit(1)

    # Load sample counts for threshold scaling
    sample_counts = _load_profile_sample_counts(pdir)

    # Load temporal constraints for filtering
    series_ranges = _load_profile_series_ranges(pdir)

    # Discover episodes
    if args.episode:
        ep_ids = args.episode if isinstance(args.episode, list) else [args.episode]
    else:
        # Auto-discover: have clusters, no labels (unless --force/--additive)
        cluster_dir = dia_dir / "clusters"
        labels_dir = dia_dir / "labels"
        existing_labels = (
            {p.stem for p in labels_dir.glob("*.json")} if labels_dir.is_dir() else set()
        )
        ep_ids = []
        if cluster_dir.is_dir():
            for cp in sorted(cluster_dir.glob("*.npz")):
                if additive:
                    # Only episodes that HAVE existing labels
                    if cp.stem in existing_labels:
                        ep_ids.append(cp.stem)
                elif args.force or cp.stem not in existing_labels:
                    ep_ids.append(cp.stem)

    # Apply series/season filters
    series_filter = getattr(args, "series", "all")
    season_filter = getattr(args, "season", None)
    if series_filter and series_filter != "all":
        code = series_filter.upper()
        ep_ids = [e for e in ep_ids if e.startswith(code + ".")]
    if season_filter is not None:
        pattern = f"S{season_filter:02d}E"
        ep_ids = [e for e in ep_ids if pattern in e]

    if args.limit:
        ep_ids = ep_ids[:args.limit]

    if not ep_ids and not merge:
        print("No episodes to auto-label.")
        return

    total = len(ep_ids)
    auto_applied = 0
    needs_review = 0

    for idx, ep_id in enumerate(ep_ids, 1):
        cluster_path = dia_dir / "clusters" / f"{ep_id}.npz"
        if not cluster_path.exists():
            print(f"[{idx:03d}/{total}] {ep_id} -- SKIP (no clusters)")
            continue

        cluster_data = np.load(str(cluster_path), allow_pickle=False)
        meta_str = str(cluster_data.get("_meta", "{}"))
        meta = json.loads(meta_str)
        season = meta.get("season", 0)
        title = meta.get("title", "")

        # Season-filter profiles
        profiles = _select_season_profiles(all_profiles, season)
        if not profiles:
            print(f"[{idx:03d}/{total}] {ep_id} -- SKIP (no profiles for season {season})")
            continue

        # Filter by series_range constraints (skip profiles for characters
        # that don't exist in this series or outside their episode range)
        if series_ranges:
            ep_series, ep_season_ep = ep_id.split(".", 1)
            profiles = {
                n: v for n, v in profiles.items()
                if n not in series_ranges  # unconstrained = match all
                or (
                    ep_series in series_ranges[n]
                    and ep_season_ep >= series_ranges[n][ep_series].get("first", "")
                    and (
                        "last" not in series_ranges[n][ep_series]
                        or ep_season_ep <= series_ranges[n][ep_series]["last"]
                    )
                )
            }
            if not profiles:
                print(f"[{idx:03d}/{total}] {ep_id} -- SKIP (no eligible profiles)")
                continue

        profile_names = list(profiles.keys())
        centroids = np.stack([profiles[n] for n in profile_names])
        c_norms = np.linalg.norm(centroids, axis=1, keepdims=True)
        c_normed = centroids / np.maximum(c_norms, 1e-10)

        # Find all cluster keys
        cluster_keys = sorted(
            k.replace("_embeddings", "")
            for k in cluster_data.files
            if k.endswith("_embeddings")
        )

        print(f"\n[{idx:03d}/{total}] {ep_id} {title}")

        # In additive mode, load existing label and only process skipped clusters
        existing_label_data: dict | None = None
        if additive:
            label_path = dia_dir / "labels" / f"{ep_id}.json"
            if label_path.exists():
                existing_label_data = json.loads(
                    label_path.read_text(encoding="utf-8"))
                skipped_set = set(existing_label_data.get("skipped", []))
                clusters_to_process = [c for c in cluster_keys if c in skipped_set]
                if not clusters_to_process:
                    print("  (no skipped clusters to process)")
                    continue
            else:
                clusters_to_process = cluster_keys
        else:
            clusters_to_process = cluster_keys

        proposed_map: dict[str, str] = {}
        review_clusters: list[str] = []

        for cluster in clusters_to_process:
            emb_key = f"{cluster}_embeddings"
            embeddings = cluster_data[emb_key]
            if len(embeddings) == 0:
                continue

            # Compute cluster centroid
            centroid = _compute_centroid(embeddings)
            centroid_normed = centroid / max(np.linalg.norm(centroid), 1e-10)

            # Cosine similarity against all profiles
            sims = centroid_normed @ c_normed.T
            ranked = sorted(
                zip(profile_names, sims.tolist()),
                key=lambda x: -x[1],
            )
            best_name, best_sim = ranked[0]
            second_name, second_sim = ranked[1] if len(ranked) > 1 else ("", 0.0)

            # Load sample dialogue from report if available
            n_segs = meta.get("clusters", {}).get(cluster, {}).get("n_segments", "?")

            # Effective threshold: per-character override > sample-scaled > base
            if best_name in char_thresholds:
                eff_threshold = char_thresholds[best_name]
            else:
                n_samples = sample_counts.get(best_name, 0)
                if n_samples < SMALL_PROFILE_SAMPLES:
                    penalty = SMALL_PROFILE_PENALTY * (1 - n_samples / SMALL_PROFILE_SAMPLES)
                else:
                    penalty = 0.0
                eff_threshold = threshold + penalty

            if best_sim >= eff_threshold:
                margin = best_sim - second_sim
                marker = "AUTO" if margin >= AUTO_LABEL_MARGIN else "auto"
                proposed_map[cluster] = best_name
                thresh_note = (f" thr={eff_threshold:.2f}"
                               if eff_threshold != threshold else "")
                print(f"  {cluster} ({n_segs} segs) -> {best_name} "
                      f"[{marker}] sim={best_sim:.3f}{thresh_note} "
                      f"(2nd: {second_name} {second_sim:.3f})")
            else:
                review_clusters.append(cluster)
                thresh_note = (f" thr={eff_threshold:.2f}"
                               if eff_threshold != threshold else "")
                print(f"  {cluster} ({n_segs} segs) -> ??? "
                      f"[REVIEW] best={best_name} sim={best_sim:.3f}{thresh_note} "
                      f"(2nd: {second_name} {second_sim:.3f})")

        if review_clusters:
            needs_review += 1

        if apply and proposed_map:
            # Write label file only (no profile merge)
            labels_dir = dia_dir / "labels"
            labels_dir.mkdir(parents=True, exist_ok=True)
            label_path = labels_dir / f"{ep_id}.json"

            if additive and existing_label_data:
                # Merge new matches into existing label file
                label_data = existing_label_data
                label_data["speaker_map"].update(proposed_map)
                existing_skipped = set(label_data.get("skipped", []))
                existing_skipped -= set(proposed_map.keys())
                label_data["skipped"] = sorted(existing_skipped,
                    key=lambda s: int(s.split("_")[1]) if "_" in s else 0)
                label_data["timestamp"] = datetime.now().isoformat(
                    timespec="seconds")
            else:
                label_data = {
                    "episode_id": ep_id,
                    "title": title,
                    "season": season,
                    "speaker_map": proposed_map,
                    "skipped": sorted(set(cluster_keys) - set(proposed_map)),
                    "auto_labeled": True,
                    "threshold": threshold,
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                }
            label_path.write_text(
                json.dumps(label_data, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            auto_applied += 1
            print(f"  -> {'Additive: ' if additive else ''}Applied "
                  f"{len(proposed_map)} labels, "
                  f"{len(review_clusters)} need review")

    # Merge labels into voice profiles (separate step)
    if merge:
        labels_dir = dia_dir / "labels"
        cluster_dir = dia_dir / "clusters"
        if not labels_dir.is_dir():
            print("No labels directory found.", file=sys.stderr)
            sys.exit(1)

        # Find label files matching the episode filter
        merge_eps = ep_ids if apply else []
        if not merge_eps:
            # Discover from existing labels
            for lp in sorted(labels_dir.glob("*.json")):
                merge_eps.append(lp.stem)
            # Apply same filters
            if series_filter and series_filter != "all":
                code = series_filter.upper()
                merge_eps = [e for e in merge_eps if e.startswith(code + ".")]
            if season_filter is not None:
                pattern = f"S{season_filter:02d}E"
                merge_eps = [e for e in merge_eps if pattern in e]

        merged_count = 0
        for ep_id in merge_eps:
            label_path = labels_dir / f"{ep_id}.json"
            cluster_path = cluster_dir / f"{ep_id}.npz"
            if not label_path.exists() or not cluster_path.exists():
                continue

            label_data = json.loads(label_path.read_text(encoding="utf-8"))
            speaker_map = label_data.get("speaker_map", {})
            if not speaker_map:
                continue

            cluster_data = np.load(str(cluster_path), allow_pickle=False)
            meta_str = str(cluster_data.get("_meta", "{}"))
            meta = json.loads(meta_str)
            season = meta.get("season", 0)

            for cluster, character in sorted(speaker_map.items()):
                emb_key = f"{cluster}_embeddings"
                if emb_key not in cluster_data.files:
                    continue
                new_embeddings = cluster_data[emb_key]
                if len(new_embeddings) == 0:
                    continue
                profile_name = _get_profile_name(character, season)
                profile_path = pdir / f"{profile_name}.npz"
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
                else:
                    all_embeddings = new_embeddings
                    all_metadata = new_metadata

                new_centroid = _compute_centroid(all_embeddings)
                _save_profile(
                    profile_path, new_centroid, all_embeddings, all_metadata)

            merged_count += 1

        if merged_count > 0:
            _rebuild_index(pdir)
            print(f"\nMerged profiles from {merged_count} episodes")

    print(f"\n{'=' * 60}")
    print(f"Auto-label: {total} episodes, "
          f"{auto_applied} applied, {needs_review} need review")
    ct_info = (f", char overrides: {char_thresholds}"
               if char_thresholds else "")
    print(f"Profiles: {len(all_profiles)} loaded, "
          f"threshold={threshold:.2f}{ct_info}")
