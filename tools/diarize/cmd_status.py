"""Status subcommand: show pipeline progress. Also provides logging setup."""

import argparse
import io
import logging
import sys
from pathlib import Path

from .config import PROGRESS_FILE
from .discovery import discover_transcripts, match_episodes, scan_videos

logger = logging.getLogger("whisperx_diarize")


def setup_logging(output_dir: Path) -> None:
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(output_dir / "whisperx_diarize.log", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(sh)


def discover_pipeline_status(
    root: Path, dia_dir: Path, video_index: dict | None = None,
) -> list[dict]:
    """Check pipeline stage completion for every transcript."""
    transcripts = discover_transcripts(root)
    if video_index:
        match_episodes(transcripts, video_index)

    jsons = {p.stem for p in dia_dir.glob("*.json") if p.stem != PROGRESS_FILE}
    cluster_dir = dia_dir / "clusters"
    npzs = {p.stem for p in cluster_dir.glob("*.npz")} if cluster_dir.is_dir() else set()
    label_dir = dia_dir / "labels"
    labels = {p.stem for p in label_dir.glob("*.json")} if label_dir.is_dir() else set()

    results = []
    for ep in transcripts:
        eid = ep.episode_id
        results.append({
            "episode_id": eid,
            "series": ep.series,
            "season": ep.season,
            "title": ep.title,
            "has_video": ep.video_path is not None,
            "has_json": eid in jsons,
            "has_clusters": eid in npzs,
            "has_labels": eid in labels,
        })
    return results


def cmd_status(args: argparse.Namespace) -> None:
    """Show pipeline progress across all episodes."""
    if sys.stdout.encoding != "utf-8":
        sys.stdout = io.TextIOWrapper(
            sys.stdout.buffer, encoding="utf-8", errors="replace")

    root = Path(__file__).resolve().parent.parent.parent
    dia_dir = root / args.diarization_dir

    # Try video scan (graceful skip if drives offline)
    video_index = None
    try:
        video_index = scan_videos(args.video_dirs)
    except Exception:
        pass

    all_eps = discover_pipeline_status(root, dia_dir, video_index)

    # Apply filters
    if args.series != "all":
        code = args.series.upper()
        all_eps = [e for e in all_eps if e["series"] == code]
    if args.season is not None:
        all_eps = [e for e in all_eps if e["season"] == args.season]

    if not all_eps:
        print("No episodes found matching filters.")
        return

    total = len(all_eps)
    n_video = sum(1 for e in all_eps if e["has_video"])
    n_json = sum(1 for e in all_eps if e["has_json"])
    n_clusters = sum(1 for e in all_eps if e["has_clusters"])
    n_labels = sum(1 for e in all_eps if e["has_labels"])
    n_complete = sum(1 for e in all_eps if e["has_labels"])

    # Header
    series_label = args.series.upper() if args.series != "all" else "all series"
    season_label = f" S{args.season:02d}" if args.season else ""
    print(f"\nPipeline Status ({series_label}{season_label}, {total} episodes)")
    print("=" * 60)
    print(f"{'Stage':<14} {'Done':>6}  {'Pending':>7}")
    print(f"{'video':.<14} {n_video:>6}  {total - n_video:>7}")
    print(f"{'process':.<14} {n_json:>6}  {n_video - n_json:>7}  (of {n_video} with video)")
    print(f"{'clusters':.<14} {n_clusters:>6}  {n_json - n_clusters:>7}  (of {n_json} processed)")
    print(f"{'labels':.<14} {n_labels:>6}  {n_clusters - n_labels:>7}  (of {n_clusters} clustered)")
    print(f"\nFully complete: {n_complete}/{total} ({100*n_complete/total:.1f}%)")

    # Filtered list based on --needs
    needs = getattr(args, "needs", None)
    if needs:
        if needs == "process":
            pending = [e for e in all_eps if e["has_video"] and not e["has_json"]]
            label = "needing processing"
        elif needs == "clusters":
            pending = [e for e in all_eps if e["has_json"] and not e["has_clusters"]]
            label = "needing clustering"
        elif needs == "labels":
            pending = [e for e in all_eps if e["has_clusters"] and not e["has_labels"]]
            label = "needing labels"
        else:  # "any"
            pending = [e for e in all_eps
                       if not e["has_labels"] and e["has_video"]]
            label = "incomplete (with video)"

        if pending:
            print(f"\nEpisodes {label} ({len(pending)}):")
            for e in pending[:50]:
                print(f"  {e['episode_id']}  {e['title']}")
            if len(pending) > 50:
                print(f"  ... and {len(pending) - 50} more")
        else:
            print(f"\nNo episodes {label}.")
