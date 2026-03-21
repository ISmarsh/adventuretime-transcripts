"""Segment-level rescoring: compare individual embeddings against profile centroids."""

from collections import Counter, defaultdict
from pathlib import Path

from .config import MAX_EMBED_DURATION, MIN_EMBED_DURATION
from .profiles import _load_all_profiles
from .speakers import _canon


def _reconstruct_embedded_segments(segments: list[dict]) -> dict[str, list[dict]]:
    """Reconstruct which segments were embedded per cluster.

    Replays the duration filter from embed-clusters to map
    embedding indices to segment timestamps.
    """
    by_cluster: dict[str, list[dict]] = defaultdict(list)
    for seg in segments:
        spk = seg.get("speaker", "")
        if not spk:
            continue
        dur = seg["end"] - seg["start"]
        if MIN_EMBED_DURATION <= dur <= MAX_EMBED_DURATION:
            by_cluster[spk].append(seg)
    return dict(by_cluster)


def rescore_disagrees(
    episode_id: str,
    disagrees: list[dict],
    segments: list[dict],
    dia_dir: Path,
) -> list[dict]:
    """Rescore disagree lines against profile centroids.

    For each disagree, finds the segment's individual embedding
    and compares against both the transcript speaker's and
    diarization speaker's profiles.

    Returns enriched disagree dicts with rescore fields.
    """
    import numpy as np

    cluster_path = dia_dir / "clusters" / f"{episode_id}.npz"
    if not cluster_path.exists():
        return disagrees

    profile_dir = dia_dir / "voice_profiles"
    profiles = _load_all_profiles(profile_dir)
    if not profiles:
        return disagrees

    # Build canon -> profile name + centroid mapping
    canon_profiles: dict[str, list[tuple[str, any]]] = defaultdict(list)
    for pname, centroid in profiles.items():
        canon_profiles[_canon(pname)].append((pname, centroid))

    cluster_data = np.load(str(cluster_path), allow_pickle=False)
    embedded_segs = _reconstruct_embedded_segments(segments)

    results = []
    for d in disagrees:
        entry = dict(d)
        cluster = d["cluster"]
        emb_key = f"{cluster}_embeddings"

        if emb_key not in cluster_data:
            results.append(entry)
            continue

        embeddings = cluster_data[emb_key]
        segs_in_cluster = embedded_segs.get(cluster, [])

        # Find embedding matching this disagree's timestamp
        d_start = d.get("w_start", -1)
        d_end = d.get("w_end", -1)

        seg_emb = None
        if d_start >= 0:
            for idx, seg in enumerate(segs_in_cluster):
                if idx >= len(embeddings):
                    break
                if seg["start"] <= d_end and seg["end"] >= d_start:
                    seg_emb = embeddings[idx]
                    entry["rescore_type"] = "segment"
                    break

        # Fall back to cluster centroid
        if seg_emb is None:
            cent_key = f"{cluster}_centroid"
            if cent_key in cluster_data:
                seg_emb = cluster_data[cent_key]
                entry["rescore_type"] = "centroid"

        if seg_emb is None:
            results.append(entry)
            continue

        emb_norm = np.linalg.norm(seg_emb)
        if emb_norm < 1e-10:
            results.append(entry)
            continue
        normed = seg_emb / emb_norm

        transcript_canon = _canon(d["transcript"])
        diarization_canon = _canon(d["diarization"])

        t_score, _t_profile = None, None
        d_score, _d_profile = None, None

        for canon_name, entries in canon_profiles.items():
            for pname, centroid in entries:
                c_norm = np.linalg.norm(centroid)
                if c_norm < 1e-10:
                    continue
                sim = float(normed @ (centroid / c_norm))
                if canon_name == transcript_canon:
                    if t_score is None or sim > t_score:
                        t_score, _t_profile = sim, pname
                if canon_name == diarization_canon:
                    if d_score is None or sim > d_score:
                        d_score, _d_profile = sim, pname

        if t_score is not None:
            entry["rescore_transcript"] = round(t_score, 3)
        if d_score is not None:
            entry["rescore_diarization"] = round(d_score, 3)

        if t_score is not None and d_score is not None:
            if t_score > d_score:
                entry["rescore_verdict"] = "transcript"
            elif d_score > t_score:
                entry["rescore_verdict"] = "diarization"
            else:
                entry["rescore_verdict"] = "tie"

        results.append(entry)

    return results


def split_clusters(
    episode_id: str,
    segments: list[dict],
    dia_dir: Path,
    label_map: dict[str, str],
) -> dict[str, dict]:
    """Analyze whether cluster segments should be split across profiles.

    For each labeled cluster, compares individual segment embeddings against
    all profile centroids and reports affiliation breakdown.

    Returns {cluster_id: {assigned, segments, splits, mixed_pct}}.
    """
    import numpy as np

    cluster_path = dia_dir / "clusters" / f"{episode_id}.npz"
    if not cluster_path.exists():
        return {}

    profile_dir = dia_dir / "voice_profiles"
    profiles = _load_all_profiles(profile_dir)
    if not profiles:
        return {}

    cluster_data = np.load(str(cluster_path), allow_pickle=False)

    # Build profile matrix for vectorized comparison
    pnames = sorted(profiles.keys())
    centroids = np.stack([profiles[p] for p in pnames])
    c_norms = np.linalg.norm(centroids, axis=1, keepdims=True)
    centroids_normed = centroids / np.maximum(c_norms, 1e-10)
    canon_pnames = [_canon(p) for p in pnames]

    results = {}
    for cluster, character in label_map.items():
        emb_key = f"{cluster}_embeddings"
        if emb_key not in cluster_data:
            continue

        embeddings = cluster_data[emb_key]
        if len(embeddings) == 0:
            continue

        e_norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embs_normed = embeddings / np.maximum(e_norms, 1e-10)
        sim_matrix = embs_normed @ centroids_normed.T  # (n_segs, n_profiles)

        best_indices = sim_matrix.argmax(axis=1)
        counts = Counter()
        for idx in best_indices:
            counts[canon_pnames[idx]] += 1

        assigned = _canon(character)
        n_segs = len(embeddings)
        non_assigned = n_segs - counts.get(assigned, 0)
        mixed_pct = round(100 * non_assigned / n_segs) if n_segs > 0 else 0

        if mixed_pct > 0:
            results[cluster] = {
                "assigned": assigned,
                "segments": n_segs,
                "splits": dict(counts.most_common()),
                "mixed_pct": mixed_pct,
            }

    return results
